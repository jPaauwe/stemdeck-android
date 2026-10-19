#!/usr/bin/env python3
"""The Android analysis stages against desktop StemDeck's own Python.

Runs each stage of the app's C++ port (analysis/, through the host tool
sdanalysis_cli) and the upstream function it ports on the same audio, and
prints how far apart they are. Exits non-zero when a difference exceeds the
limits in the release notes. Like scripts/engine_test.py, it needs the real
model files; CI downloads them.

  python3 scripts/analysis_parity.py --cli build/analysis/sdanalysis_cli \\
      --upstream <stemdeck checkout> --beat-ckpt final0.ckpt --beat-onnx final0.onnx \\
      --ort-lib libonnxruntime.so --audio <files...>

The upstream checkout must be the commit in app/src/main/assets/web/UPSTREAM_COMMIT,
with its Python dependencies installed (CI uses its uv.lock).
"""

from __future__ import annotations

import argparse
import shutil
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 22050


# ── test material ─────────────────────────────────────────────────────


def synth_band(bpm: float, seconds: float, swing: float = 0.0, seed: int = 0, sr: int = 44100) -> np.ndarray:
    """A drum kit (kick, snare, hats) plus bass at `bpm`, with a little
    human timing, as stereo float. Deterministic."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros(n)
    beat = 60.0 / bpm
    t = 0.4
    i = 0

    def hit(at, kind, gain=1.0):
        s = int(at * sr)
        if s >= n:
            return
        if kind == "kick":
            ln = int(0.25 * sr)
            tt = np.arange(ln) / sr
            sig = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
        elif kind == "snare":
            ln = int(0.18 * sr)
            tt = np.arange(ln) / sr
            sig = (rng.standard_normal(ln) * 0.6 + np.sin(2 * np.pi * 190 * tt)) * np.exp(-tt * 25)
        else:  # hat
            ln = int(0.05 * sr)
            tt = np.arange(ln) / sr
            sig = rng.standard_normal(ln) * np.exp(-tt * 90) * 0.35
        e = min(n, s + len(sig))
        out[s:e] += gain * sig[: e - s]

    while t < seconds - 0.5:
        jitter = rng.normal(0, 0.004)
        at = t + jitter
        hit(at, "kick" if i % 2 == 0 else "snare")
        hit(at, "hat", 0.8)
        hit(at + beat / 2 + swing * beat, "hat", 0.5)
        t += beat
        i += 1
    peak = np.abs(out).max() or 1.0
    out = 0.8 * out / peak
    return np.stack([out, out * 0.97], axis=1)


def synth_click(bpm: float, seconds: float, sr: int = 44100) -> np.ndarray:
    n = int(seconds * sr)
    out = np.zeros(n)
    ln = int(0.03 * sr)
    tt = np.arange(ln) / sr
    click = np.sin(2 * np.pi * 1000 * tt) * np.exp(-tt * 120)
    t = 0.5
    while t < seconds - 0.1:
        s = int(t * sr)
        out[s : s + ln] += click[: max(0, min(ln, n - s))]
        t += 60.0 / bpm
    return np.stack([out, out], axis=1) * 0.7


def write_wav16(path: Path, x: np.ndarray, sr: int = 44100) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def to_stem_wav(src: str, dest: Path) -> None:
    """What the engine writes: 16-bit stereo 44.1 kHz."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-ac", "2", "-ar", "44100",
         "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


# ── reference side ────────────────────────────────────────────────────


def upstream_env(upstream: Path, data_dir: Path, beat_ckpt: str) -> None:
    os.environ["STEMDECK_DATA_DIR"] = str(data_dir)
    os.environ["STEMDECK_JOBS_DIR"] = str(data_dir / "jobs")
    os.environ["STEMDECK_BEAT_MODEL"] = beat_ckpt
    os.environ["STEMDECK_BEAT_DETECTOR"] = "model"
    sys.path.insert(0, str(upstream))


def run(cmd: list[str], env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1]} failed: {r.stderr.strip()}")
    return r.stderr.strip()


# ── comparisons ───────────────────────────────────────────────────────


def compare_grids(ref: dict | None, mine: dict | None) -> dict:
    if ref is None or mine is None:
        return {"ok": ref is None and mine is None, "ref": ref is not None, "mine": mine is not None}
    rb, mb = np.asarray(ref["beats"]), np.asarray(mine["beats"])
    same_count = len(rb) == len(mb)
    if same_count and len(rb):
        err = np.abs(rb - mb)
    else:
        err = np.array([np.abs(mb - b).min() for b in rb]) if len(mb) else np.array([np.inf])
    return {
        "ok": True,
        "beats": (len(rb), len(mb)),
        "detected": (ref["detected"], mine["detected"]),
        "bars": (ref["bars"], mine["bars"]),
        "bpm": (ref["bpm"], mine["bpm"]),
        "confidence": (ref["confidence"], mine["confidence"]),
        "median_ms": float(np.median(err) * 1000),
        "max_ms": float(err.max() * 1000),
        "same_count": same_count,
        "same_bars": ref["bars"] == mine["bars"],
        "fields_equal": all(ref[k] == mine[k] for k in ref if k not in ("beats", "onsets")),
    }


# ── sections ──────────────────────────────────────────────────────────

SIX = ("bass", "drums", "vocals", "other", "guitar", "piano")


def fake_stems(src: str, dest: Path) -> None:
    """Six 16-bit stems cut from one recording with filters: not a
    separation, but real music for the model and the same files for both
    sides, which is all a parity check needs."""
    dest.mkdir(parents=True, exist_ok=True)
    filters = {
        "bass": "lowpass=f=180,lowpass=f=180", "drums": "highpass=f=4000,volume=1.5",
        "vocals": "bandpass=f=1500:width_type=h:w=2000", "other": "volume=0.4",
        "guitar": "bandpass=f=800:width_type=h:w=1200,volume=0.5", "piano": "highpass=f=300,volume=0.3",
    }
    for name, af in filters.items():
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-af", af, "-ac", "2",
                        "-ar", "44100", "-c:a", "pcm_s16le", str(dest / f"{name}.wav")], check=True)


def reference_sections(stems: Path, work: Path, ckpt_dir: str, beats_json: Path | None) -> dict:
    """What desktop's detect_sections does with these stems: the ffmpeg amix of
    other/guitar/piano, then section_worker's body (the ensemble loaded from
    local checkpoints instead of the Hub), then normalize_sections."""
    import torch
    from allin1_infer.config import HARMONIX_LABELS
    from allin1_infer.helpers import run_inference
    from allin1_infer.models.ensemble import Ensemble
    from allin1_infer.models.loaders import NAME_TO_FILE, _build_model_from_checkpoint
    from allin1_infer.spectrogram import extract_spectrograms
    from app.pipeline.section_refine import refine_segments
    from app.pipeline.section_worker import _load_beat_grid, _result_segments
    from app.pipeline.sections import normalize_sections

    wd = work / "sections-work"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    for name in ("bass", "drums", "vocals"):
        shutil.copy(stems / f"{name}.wav", wd / f"{name}.wav")
    cmd = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    for name in ("other", "guitar", "piano"):
        cmd += ["-i", str(stems / f"{name}.wav")]
    cmd += ["-filter_complex", "[0:a][1:a][2:a]amix=inputs=3:normalize=0:duration=longest", "-c:a", "pcm_f32le",
            str(wd / "other.wav")]
    subprocess.run(cmd, check=True)
    spec_path = extract_spectrograms([wd], wd / "spec", multiprocess=False)[0]
    folds = [_build_model_from_checkpoint(str(Path(ckpt_dir) / NAME_TO_FILE[f"harmonix-fold{k}"]), "cpu") for k in range(8)]
    model = Ensemble(folds).eval()
    with torch.no_grad():
        result = run_inference(path=Path("x.wav"), spec_path=spec_path, model=model, device="cpu",
                               include_activations=True, include_embeddings=True)
        logits = model(torch.from_numpy(np.load(spec_path)).unsqueeze(0))
    raw = _result_segments(result)
    refined = refine_segments(raw, result.activations, result.embeddings, result.activation_fps,
                              _load_beat_grid(beats_json), HARMONIX_LABELS)
    frames = np.load(spec_path).shape[1]
    return {
        "features": np.load(spec_path).astype(np.float32),
        "section": logits.logits_section[0].numpy().astype(np.float32),
        "function": logits.logits_function[0].numpy().astype(np.float32),
        "embedding": result.embeddings.astype(np.float64).mean(axis=-1),  # (4, T, 24), what refine reads
        "raw": raw, "refined": refined, "frames": frames,
        "normalize": lambda d: normalize_sections(refined, d),
    }


def label_overlap(a: list, b: list, duration: float) -> float:
    """Share of the song's duration where both lists name the same kind."""
    edges = sorted({0.0, duration, *(s["start"] for s in a + b), *(s["end"] for s in a + b)})
    same = 0.0
    for lo, hi in zip(edges, edges[1:]):
        mid = (lo + hi) / 2
        ka = next((s["kind"] for s in a if s["start"] <= mid < s["end"]), None)
        kb = next((s["kind"] for s in b if s["start"] <= mid < s["end"]), None)
        if ka is not None and ka == kb:
            same += hi - lo
    return same / duration if duration else 1.0


def sections_stage(args, work: Path) -> list[str]:
    failures = []
    cases = [(Path(d).name + " (separated)", Path(d)) for d in args.stems]
    for a in args.audio:
        d = work / "fake-stems" / Path(a).stem
        fake_stems(a, d)
        cases.append((Path(a).name[:40] + " (filtered)", d))
    print(f"\nSections (All-In-One harmonix-all + section_refine), {len(cases)} songs:")
    print(f"{'case':52s} {'spec max':>9s} {'post':>5s} {'sections':>9s} {'bound max':>9s} {'labels':>7s}")
    for name, stems in cases:
        sd = work / "sec" / stems.name
        sd.mkdir(parents=True, exist_ok=True)
        # the beat grid both sides refine against: the port's (equal to desktop's, see the beatgrid stage)
        grid_json = sd / "beats.json"
        run([args.cli, "beatgrid", str(stems / "drums.wav"), args.beat_onnx, str(grid_json), "2"])
        grid = json.loads(grid_json.read_text())
        beats_txt = sd / "beats.txt"
        conf = -1.0
        if grid:
            beats_txt.write_text("\n".join(repr(b) for b in grid["beats"]))
            conf = float(grid["confidence"])
        ref = reference_sections(stems, work, args.allinone_dir, grid_json if grid else None)
        T = ref["frames"]
        duration = T * 441 / 44100  # what the app passes: the stems' length
        # 1. the model input
        run([args.cli, "secfeat", str(stems), str(sd / "feat.f32")])
        mine_feat = np.fromfile(sd / "feat.f32", np.float32)
        spec_err = float(np.abs(mine_feat - ref["features"].ravel()).max()) if mine_feat.size == ref["features"].size else float("inf")
        # 2. post-processing alone, from desktop's own ensemble outputs
        ref["section"].tofile(sd / "sec.f32")
        ref["function"].tofile(sd / "fun.f32")
        np.ascontiguousarray(np.transpose(ref["embedding"], (1, 0, 2)).reshape(T, -1)).astype(np.float64).tofile(sd / "emb.f64")
        run([args.cli, "secpost", str(sd / "sec.f32"), str(sd / "fun.f32"), str(sd / "emb.f64"), str(T),
             str(beats_txt) if grid else "-", str(conf), repr(duration), str(sd / "post.json")])
        post = json.loads((sd / "post.json").read_text())
        ref_sections = ref["normalize"](duration)
        post_same = (post["raw"] == ref["raw"] and post["refined"] == ref["refined"] and post["sections"] == ref_sections)
        # 3. the whole stage
        run([args.cli, "sections", str(stems), args.section_onnx, str(beats_txt) if grid else "-", str(conf),
             repr(duration), str(sd / "full.json"), "2"])
        full = json.loads((sd / "full.json").read_text())
        mine = full["sections"]
        same_count = len(mine) == len(ref_sections)
        bmax = max((abs(x["start"] - y["start"]) for x, y in zip(mine, ref_sections)), default=0.0) if same_count else float("inf")
        overlap = label_overlap(mine, ref_sections, duration) if (mine or ref_sections) else 1.0  # both found none
        print(f"{name:52s} {spec_err:9.1e} {'same' if post_same else 'DIFF':>5s} "
              f"{len(ref_sections):>4d}/{len(mine):<4d} {bmax:9.3f} {overlap:7.0%}")
        if not post_same:
            print("   desktop raw    ", ref["raw"]); print("   Android raw    ", post["raw"])
            print("   desktop refined", ref["refined"]); print("   Android refined", post["refined"])
            failures.append(f"{name}: post-processing differs given desktop's own model outputs")
        if spec_err > 1e-5:
            failures.append(f"{name}: spectrogram differs by {spec_err}")
        if not same_count or bmax > 0.1 or overlap < 0.9:
            print("   desktop", [(s["kind"], s["start"], s["end"]) for s in ref_sections])
            print("   Android", [(s["kind"], s["start"], s["end"]) for s in mine])
            failures.append(f"{name}: sections differ (count {len(ref_sections)}/{len(mine)}, boundaries {bmax:.3f} s, labels {overlap:.0%})")
        print(f"   timings: spectrogram {full['t_spec']:.2f} s, eight folds {full['t_model']:.2f} s, post {full['t_post']:.3f} s for {duration:.0f} s of audio")
    return failures


# ── lead/backing vocal split ─────────────────────────────────────────


def read_wav16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64).reshape(-1, w.getnchannels())


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    n = min(len(ref), len(x))
    err = ((ref[:n] - x[:n]) ** 2).sum()
    return float("inf") if err == 0 else float(10 * np.log10((ref[:n] ** 2).sum() / err))


def vocalsplit_stage(args, work: Path) -> list[str]:
    """audio-separator (UVR-MDX-NET Karaoke 2, desktop's vocal_split_worker
    call) against the port, on each vocals stem."""
    from audio_separator.separator import Separator

    failures = []
    cases = [(Path(d).name + " (separated)", Path(d) / "vocals.wav") for d in args.stems]
    for a in args.audio:
        d = work / "fake-stems" / Path(a).stem
        if not (d / "vocals.wav").is_file():
            fake_stems(a, d)
        cases.append((Path(a).name[:40] + " (filtered)", d / "vocals.wav"))
    print(f"\nLead/backing vocal split (UVR-MDX-NET Karaoke 2), {len(cases)} vocals stems:")
    print(f"{'case':52s} {'lead SNR':>9s} {'backing SNR':>11s} {'same samples':>12s} {'lead+1.065 backing':>19s}")
    sep = Separator(log_level=40, model_file_dir=args.karaoke_dir, output_dir=str(work / "vs-ref"), output_format="WAV")
    sep.load_model(model_filename="UVR_MDXNET_KARA_2.onnx")
    model = str(Path(args.karaoke_dir) / "UVR_MDXNET_KARA_2.onnx")
    for name, vocals in cases:
        ref_dir, my_dir = work / "vs-ref", work / "vs-mine"
        for d in (ref_dir, my_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        sep.output_dir = str(ref_dir)
        sep.model_instance.output_dir = str(ref_dir)
        sep.separate(str(vocals), {"Vocals": "lead_vocals", "Instrumental": "backing_vocals"})
        r = subprocess.run([args.cli, "vsplit", str(vocals), str(my_dir / "lead_vocals.wav"), str(my_dir / "backing_vocals.wav"), model, "2"],
                           capture_output=True, text=True)
        have_ref = all((ref_dir / f"{n}.wav").is_file() for n in ("lead_vocals", "backing_vocals"))
        if r.returncode != 0 or not have_ref:
            same = (r.returncode == 3) and not have_ref  # both found nothing to split
            print(f"{name:52s} {'(no split on ' + ('either side' if same else 'one side') + ')':>40s}")
            if not same:
                failures.append(f"{name}: vocal split ran on one side only ({r.stderr.strip()})")
            continue
        lr, br = read_wav16(ref_dir / "lead_vocals.wav"), read_wav16(ref_dir / "backing_vocals.wav")
        lm, bm = read_wav16(my_dir / "lead_vocals.wav"), read_wav16(my_dir / "backing_vocals.wav")
        same = float(np.mean(np.concatenate([(lr == lm).ravel(), (br == bm).ravel()]))) if lr.shape == lm.shape and br.shape == bm.shape else 0.0
        v = read_wav16(vocals)
        # The model's two outputs are complementary by construction: lead = vocals - 1.065 x backing
        # (each written with its own peak normalisation, so only check where none was applied).
        recon = snr_db(v, lm + 1.065 * bm) if max(np.abs(v).max(), np.abs(lm).max(), np.abs(bm).max()) < 0.9 * 32767 else float("nan")
        ls, bs = snr_db(lr, lm), snr_db(br, bm)
        print(f"{name:52s} {ls:7.1f} dB {bs:9.1f} dB {same:12.4%} {recon:16.1f} dB")
        if lr.shape != lm.shape or br.shape != bm.shape:
            failures.append(f"{name}: vocal split lengths differ")
        elif min(ls, bs) < 40:
            failures.append(f"{name}: vocal split SNR {ls:.1f}/{bs:.1f} dB (< 40)")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--beat-ckpt", required=True)
    ap.add_argument("--beat-onnx", required=True)
    ap.add_argument("--ort-lib", required=True)
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--stages", default="beatgrid")
    ap.add_argument("--keep", default="")
    ap.add_argument("--allinone-dir", help="the eight harmonix-fold*.pth checkpoints (sections stage)")
    ap.add_argument("--section-onnx", help="folder with allinone_harmonix_fold0..7.onnx (sections stage)")
    ap.add_argument("--stems", nargs="*", default=[], help="folders of real six-stem separations (sections stage)")
    ap.add_argument("--karaoke-dir", help="folder with UVR_MDXNET_KARA_2.onnx for audio-separator (vocalsplit stage)")
    args = ap.parse_args()

    work = Path(args.keep or tempfile.mkdtemp(prefix="sda-parity-"))
    data = work / "data"
    upstream_env(Path(args.upstream), data, args.beat_ckpt)
    (data / "jobs").mkdir(parents=True, exist_ok=True)
    os.environ["SDA_ORT_LIB"] = args.ort_lib

    failures: list[str] = []
    stages = set(args.stages.split(","))

    if "beatgrid" in stages:
        from app.pipeline import beatgrid as bg  # noqa: E402

        cases: list[tuple[str, Path]] = []
        synth = work / "synth"
        synth.mkdir(parents=True, exist_ok=True)
        for bpm in (72, 120, 176, 200):
            p = synth / f"band-{bpm}.wav"
            write_wav16(p, synth_band(bpm, 40, seed=bpm))
            cases.append((f"synthetic band {bpm} BPM", p))
        for bpm in (60, 90, 140, 180):
            p = synth / f"click-{bpm}.wav"
            write_wav16(p, synth_click(bpm, 30))
            cases.append((f"click {bpm} BPM", p))
        for a in args.audio:
            cases.append((Path(a).name, Path(a)))

        print(f"{'case':58s} {'beats ref/port':>15s} {'bars equal':>10s} {'BPM ref/port':>20s} {'median':>8s} {'max':>8s}")
        for i, (name, src) in enumerate(cases):
            job = f"{i:012x}"
            stems = data / "jobs" / job / "stems"
            stems.mkdir(parents=True, exist_ok=True)
            drums = stems / "drums.wav"
            to_stem_wav(str(src), drums)
            ref = bg.compute_beat_grid(stems)
            out = work / f"{job}.json"
            run([args.cli, "beatgrid", str(drums), args.beat_onnx, str(out)])
            mine = json.loads(out.read_text())
            c = compare_grids(ref, mine)
            if not c["ok"]:
                print(f"{name[:58]:58s} grid ref={c['ref']} port={c['mine']}")
                failures.append(f"{name}: one side has no grid")
                continue
            if ref is None:
                print(f"{name[:58]:58s} no grid on either side")
                continue
            print(
                f"{name[:58]:58s} {c['beats'][0]:>7d}/{c['beats'][1]:<7d} {str(c['same_bars']):>10s} "
                f"{c['bpm'][0]:>9.3f}/{c['bpm'][1]:<10.3f} {c['median_ms']:>6.2f}ms {c['max_ms']:>6.1f}ms"
            )
            if not c["same_count"]:
                failures.append(f"{name}: {c['beats'][0]} beats on desktop, {c['beats'][1]} on Android")
            elif c["median_ms"] >= 5.0:
                failures.append(f"{name}: median beat error {c['median_ms']:.2f} ms")
            if not c["same_bars"]:
                failures.append(f"{name}: bar marks differ {c['bars']}")
            # The clean-up alone, given desktop's own detection: must agree to
            # float rounding.
            y = np.frombuffer(
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(drums), "-ac", "1", "-ar", str(SR),
                     "-f", "f32le", "-"], capture_output=True, check=True).stdout, dtype=np.float32)
            got = {}
            orig = bg.detect_beats

            def capture(yy, sr, env):
                b, d, det = orig(yy, sr, env)
                got["b"], got["d"] = b, d
                return b, d, det

            bg.detect_beats = capture
            try:
                ref2 = bg.compute_beat_grid(stems)
            finally:
                bg.detect_beats = orig
            if ref2 is not None:
                (work / "y.f32").write_bytes(y.tobytes())
                (work / "b.txt").write_text("\n".join(repr(float(t)) for t in got["b"]))
                (work / "d.txt").write_text("\n".join(repr(float(t)) for t in got["d"]))
                run([args.cli, "grid", str(work / "y.f32"), str(work / "b.txt"), str(work / "d.txt"), str(work / "g.json")])
                g2 = json.loads((work / "g.json").read_text())
                c2 = compare_grids(ref2, g2)
                if not (c2["ok"] and c2["same_count"] and c2["max_ms"] < 0.01 and c2["fields_equal"]):
                    diffs = {k: (ref2[k], g2[k]) for k in ref2 if k not in ("beats", "onsets") and ref2[k] != g2[k]}
                    print(f"   clean-up alone differs: {c2.get('max_ms')} ms, {diffs}")
                    failures.append(f"{name}: clean-up port differs given the same detection")
                if ref2["onsets"] != g2["onsets"]:
                    ro, mo = np.asarray(ref2["onsets"]), np.asarray(g2["onsets"])
                    print(f"   onsets: {len(ro)} vs {len(mo)}")

    if "key" in stages:
        from app.core.models import Job  # noqa: E402
        from app.pipeline import analyze as az  # noqa: E402
        from tests import keyset  # noqa: E402

        kdir = work / "key"
        kdir.mkdir(parents=True, exist_ok=True)
        songs = keyset.songs()
        same = 0
        confusions = []
        print(f"\nKey set (tests/keyset.py, {len(songs)} songs, stems path):")
        for song in songs:
            h = song.harmony.astype(np.float32)
            b = song.bass.astype(np.float32)
            ref = az.detect_key_from_audio(h, b, keyset.SR)
            (kdir / "h.f32").write_bytes(h.tobytes())
            (kdir / "b.f32").write_bytes(b.tobytes())
            out = subprocess.run([args.cli, "key", str(kdir / "h.f32"), str(kdir / "b.f32"), "1"],
                                 capture_output=True, text=True, check=True).stdout
            mine = json.loads(out)
            got = (mine["key"], mine["scale"], mine["confidence"])
            if got[:2] == tuple(ref[:2]):
                same += 1
            else:
                confusions.append(f"{song.name} in {song.label}: desktop {ref[0]} {ref[1]}, Android {got[0]} {got[1]}")
            if got != tuple(ref):
                print(f"   {song.name:14s} {song.label:6s} desktop {ref}  Android {got}")
        share = same / len(songs)
        print(f"   same key and scale on {same} of {len(songs)} ({share:.0%})")
        for c in confusions:
            print("   confusion:", c)
        if share < 0.98:
            failures.append(f"key set: {same} of {len(songs)} agree")

        # Real recordings through the stems path (the recording as the
        # "other" stem, its low end as the bass stem).
        for i, a in enumerate(args.audio):
            job = Job(id=f"{0xabc000 + i:012x}")
            stems = data / "jobs" / job.id / "stems"
            stems.mkdir(parents=True, exist_ok=True)
            to_stem_wav(a, stems / "other.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", a, "-af", "lowpass=f=200,lowpass=f=200",
                            "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(stems / "bass.wav")], check=True)
            job.key = job.scale = None
            az.refine_key_from_stems(job, stems)
            out = subprocess.run([args.cli, "keystems", str(stems)], capture_output=True, text=True, check=True).stdout
            mine = json.loads(out)
            ref = (job.key, job.scale, job.key_confidence)
            got = (mine["key"], mine["scale"], mine["confidence"])
            flag = "" if got[:2] == ref[:2] else "   <-- differs"
            print(f"   {Path(a).name[:50]:50s} desktop {ref}  Android {got}{flag}")
            if got[:2] != ref[:2]:
                failures.append(f"{Path(a).name}: key {ref[:2]} on desktop, {got[:2]} on Android")

    if "click" in stages:
        from app.pipeline import click_render as cr  # noqa: E402

        fixture = json.loads((Path(__file__).resolve().parent.parent / "analysis" / "test" / "fixtures" / "click_levels.json").read_text())
        bad = 0
        for n, want in fixture["defaultGrouping"].items():
            got = subprocess.run([args.cli, "defgroup", n], capture_output=True, text=True, check=True).stdout.split()
            bad += [int(x) for x in got] != want
        for e in fixture["levels"]:
            n = e["beatsPerBar"]
            groups = "+".join(map(str, e["groups"])) if e.get("groups") else "-"
            cmd = [args.cli, "levels", str(n), groups, str(len(e["levels"]))]
            if n == -1:
                cmd.append(",".join(f"{b['beat']}:{b['beats_per_bar']}" for b in e["bars"]))
            got = [int(x) for x in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()]
            if got != e["levels"]:
                bad += 1
                print(f"   levels {n} {groups}: got {got}, want {e['levels']}")
        total = len(fixture["defaultGrouping"]) + len(fixture["levels"])
        print(f"\nClick levels against analysis/test/fixtures/click_levels.json: {total - bad} of {total} agree")
        if bad:
            failures.append(f"click levels: {bad} differ from the fixture")
        # The rendered click, sample for sample, against click_render.py.
        grids = [
            ([0.5 + i * 0.5 for i in range(40)], [{"beat": 0, "beats_per_bar": 4}], 21.0),
            ([0.31 + i * 0.4137 + 0.01 * math.sin(i) for i in range(80)],
             [{"beat": 2, "beats_per_bar": 7}, {"beat": 30, "beats_per_bar": 6}], 34.2),
            ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 33.98, 34.0], [], 34.0),
        ]
        cdir = work / "click"
        cdir.mkdir(parents=True, exist_ok=True)
        worst = 0
        cases = 0
        for gi, (beats, bars, dur) in enumerate(grids):
            (cdir / "b.txt").write_text("\n".join(repr(float(b)) for b in beats))
            bars_arg = ",".join(f"{b['beat']}:{b['beats_per_bar']}" for b in bars) or "-"
            for mult in (0.5, 1.0, 2.0):
                for accent in (-1, 0, 3, 7):
                    ref_path = cdir / "ref.wav"
                    cr.render_click_wav(ref_path, beats, bars, dur, 44100, mult, accent)
                    with wave.open(str(ref_path)) as w:
                        ref = np.frombuffer(w.readframes(w.getnframes()), "<i2")
                    run([args.cli, "clickwav", str(cdir / "b.txt"), bars_arg, repr(dur), repr(mult), str(accent),
                         str(cdir / "mine.s16"), "44100"])
                    mine = np.fromfile(cdir / "mine.s16", "<i2")
                    cases += 1
                    if len(ref) != len(mine):
                        failures.append(f"click render {gi} x{mult} accent {accent}: {len(ref)} vs {len(mine)} samples")
                        continue
                    worst = max(worst, int(np.abs(ref.astype(int) - mine.astype(int)).max()))
        print(f"Click render against click_render.py: {cases} renders, largest sample difference {worst} (16-bit steps)")
        if worst > 1:
            failures.append(f"click render differs by up to {worst} steps")

    if "sections" in stages:
        failures += sections_stage(args, work)

    if "vocalsplit" in stages:
        failures += vocalsplit_stage(args, work)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll within limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
