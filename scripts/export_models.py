#!/usr/bin/env python3
"""Exports the analysis models to ONNX from their published checkpoints.

The app runs these with ONNX Runtime on the phone. Each export is checked
against the PyTorch model it came from before it is written, and its SHA-256
must match models/models.json: with the pinned versions in
.github/workflows/models.yml the export is reproducible byte for byte, so CI
can rebuild and check a published file at any time.

  python3 scripts/export_models.py --out models-out \\
      --beat-this-ckpt final0.ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

MANIFEST = Path(__file__).resolve().parent.parent / "models" / "models.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_beat_this(ckpt: str, out: Path) -> None:
    """Beat This! (github.com/CPJKU/beat_this, MIT): spect (1, frames, 128) ->
    beat, downbeat logits (1, frames). Frames are dynamic; the app feeds the
    1500-frame chunks split_predict_aggregate uses."""
    from beat_this.inference import load_model

    model = load_model(ckpt, "cpu")
    # The rotary embedding caches its frequencies per sequence length, which
    # would bake one length into the graph.
    for m in model.modules():
        if m.__class__.__name__ == "RotaryEmbedding":
            m.cache_if_possible = False
    model.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spect):
            o = self.m(spect)
            return o["beat"], o["downbeat"]

    w = Wrap(model).eval()
    torch.onnx.export(
        w, (torch.randn(1, 1500, 128),), str(out),
        input_names=["spect"], output_names=["beat", "downbeat"],
        dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}},
        # No constant folding: folding computes constants with this machine's
        # torch kernels (AVX2 vs AVX-512 differ in the last bits), which made
        # the file's bytes depend on the CPU it was exported on.
        opset_version=17, do_constant_folding=False, dynamo=False,
    )
    import onnxruntime as ort

    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(0)
    for frames in (1500, 1000, 300, 37):
        x = torch.randn(1, frames, 128, generator=g) * 2 + 3
        with torch.inference_mode():
            b, d = w(x)
        ob, od = s.run(None, {"spect": x.numpy()})
        err = max(float(np.abs(ob - b.numpy()).max()), float(np.abs(od - d.numpy()).max()))
        print(f"  beat_this {frames} frames: ONNX vs PyTorch max |diff| {err:.2e}")
        if err > 1e-4:
            raise SystemExit("beat_this export does not match PyTorch")


ALLINONE_FOLDS = [
    "harmonix-fold0-0vra4ys2.pth", "harmonix-fold1-3ozjhtsj.pth", "harmonix-fold2-gmgo0nsy.pth",
    "harmonix-fold3-i92b7m8p.pth", "harmonix-fold4-1bql5qo0.pth", "harmonix-fold5-x4z5zeef.pth",
    "harmonix-fold6-x7t226rq.pth", "harmonix-fold7-qwwskhg6.pth",
]


def export_allinone(ckpt_dir: str, out_dir: Path) -> list[Path]:
    """All-In-One (github.com/mir-aidj/all-in-one, MIT), the harmonix-all
    ensemble desktop runs, one ONNX file per fold: features (1, 4, frames, 81)
    -> beat, downbeat, section (1, frames), function (1, 10, frames),
    embedding (1, 4, frames, 24). The app averages the folds like
    allin1_infer.models.ensemble.Ensemble. Built from all-in-one-infer's
    pure-PyTorch model (its neighbourhood attention replaces NATTEN)."""
    from omegaconf import OmegaConf
    from allin1_infer.models import dinat, neighborhood_attention as na
    from allin1_infer.models.allinone import AllInOne

    # Two things in the reference bake the traced length into the graph: the
    # neighbour index tables are built from a Python int (and lru_cached), and
    # the pad below a layer's window is a Python `if`. Both are replaced, for
    # the export only, by tensor arithmetic with the same values: indices from
    # the traced length, and the pad as a gather that appends exactly
    # max(0, window - frames) zeros (a no-op when frames >= window).
    def na1d_indices(length, kernel_size, dilation, device):
        i = torch.arange(length, device=device)
        q_class = i // dilation
        residue = i % dilation
        class_len = (length - residue + dilation - 1) // dilation
        start = (q_class - (kernel_size - 1) // 2).clamp(min=0)
        start = torch.minimum(start, class_len - kernel_size)
        j = torch.arange(kernel_size, device=device)
        neighbor_class = start[:, None] + j[None, :]
        return neighbor_class * dilation + residue[:, None], neighbor_class - q_class[:, None] + (kernel_size - 1)

    def maybe_pad_1d(self, hidden_states, frames):
        length = torch._shape_as_tensor(hidden_states)[1]
        idx = torch.arange(torch.clamp(length, min=self.window_size))
        keep = (idx < length).to(hidden_states.dtype)
        padded = hidden_states[:, torch.minimum(idx, length - 1)] * keep[None, :, None]
        return padded, (0, 0, 0, 1)  # always crop back to `frames` afterwards

    original = (na._na1d_indices, dinat.DinatLayer1d.maybe_pad)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features):
            o = self.m(features)
            return o.logits_beat, o.logits_downbeat, o.logits_section, o.logits_function, o.embeddings

    import onnxruntime as ort

    paths = []
    for fold, name in enumerate(ALLINONE_FOLDS):
        ck = torch.load(str(Path(ckpt_dir) / name), map_location="cpu")
        model = AllInOne(OmegaConf.create(ck["config"]))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        w = Wrap(model).eval()
        out = out_dir / f"allinone_harmonix_fold{fold}.onnx"
        names = ["beat", "downbeat", "section", "function", "embedding"]
        na._na1d_indices, dinat.DinatLayer1d.maybe_pad = na1d_indices, maybe_pad_1d
        try:
            torch.onnx.export(
                w, (torch.randn(1, 4, 1000, 81),), str(out),
                input_names=["features"], output_names=names,
                dynamic_axes={"features": {2: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"},
                              "section": {1: "frames"}, "function": {2: "frames"}, "embedding": {2: "frames"}},
                opset_version=17, do_constant_folding=False, dynamo=False,  # see export_beat_this
            )
        finally:
            na._na1d_indices, dinat.DinatLayer1d.maybe_pad = original
        # checked against the unpatched model, on both sides of the largest window (10240 frames)
        s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        g = torch.Generator().manual_seed(fold)
        worst = 0.0
        for frames in (1000, 333, 12000):
            x = torch.rand(1, 4, frames, 81, generator=g) * 2
            with torch.inference_mode():
                ref = w(x)
            got = s.run(None, {"features": x.numpy()})
            for r, o in zip(ref, got):
                if tuple(r.shape) != o.shape:
                    raise SystemExit(f"all-in-one fold {fold}: shape {o.shape} != {tuple(r.shape)}")
                worst = max(worst, float(np.abs(o - r.numpy()).max()))
        print(f"  all-in-one fold {fold}: ONNX vs PyTorch max |diff| {worst:.2e}")
        if worst > 1e-4:
            raise SystemExit("all-in-one export does not match PyTorch")
        paths.append(out)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat-this-ckpt")
    ap.add_argument("--allinone-dir", help="folder with the eight harmonix-fold*.pth checkpoints")
    ap.add_argument("--check-manifest", action="store_true", help="fail if a SHA-256 differs from models.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    produced = {}
    if args.beat_this_ckpt:
        p = out / "beat_this_final0.onnx"
        export_beat_this(args.beat_this_ckpt, p)
        produced[p.name] = p
    if args.allinone_dir:
        for p in export_allinone(args.allinone_dir, out):
            produced[p.name] = p
    bad = 0
    for name, p in produced.items():
        digest, size = sha256(p), p.stat().st_size
        want = manifest.get(name, {})
        same = want.get("sha256") == digest and want.get("bytes") == size
        print(f"{name}: {size} bytes, sha256 {digest} {'(matches models.json)' if same else '(DIFFERS from models.json)'}")
        if args.check_manifest and not same:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
