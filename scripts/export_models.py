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
        opset_version=17, do_constant_folding=True, dynamo=False,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--beat-this-ckpt")
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
