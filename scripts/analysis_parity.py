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

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll within limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
