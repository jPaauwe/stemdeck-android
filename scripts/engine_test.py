#!/usr/bin/env python3
"""CI test of the Android separation engine with the real htdemucs_6s model.

Runs the same C++ engine the app ships (built for the CI host by
app/src/main/cpp/CMakeLists.txt) on a synthetic 48 kHz test song and checks:

  * resampling to 44.1 kHz keeps the length;
  * integrated loudness matches pyloudnorm (what desktop StemDeck uses);
  * the six stems add back up to the mix;
  * separating in parallel chunks gives the same stems as one pass, which
    tests the chunk blending the phone relies on.

usage: engine_test.py <sdengine_cli> <model.bin> <workdir>
"""
import json
import os
import subprocess
import sys

import numpy as np
import pyloudnorm

CLI, MODEL, WORK = sys.argv[1:4]
os.makedirs(WORK, exist_ok=True)
SOURCES = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def last_json(out):
    """The engine library logs to stdout; the CLI's JSON result is its last line."""
    return json.loads([l for l in out.splitlines() if l.startswith("{")][-1])


def run(*args):
    print("$", " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True)
    sys.stderr.write(r.stderr[-3000:])
    if r.returncode:
        sys.exit(f"command failed: {r.returncode}")
    return r.stdout


def song(fs=48000, seconds=45):
    rng = np.random.default_rng(7)
    t = np.arange(fs * seconds) / fs
    out = np.zeros((len(t), 2))
    beat = 60 / 120
    # kick + hats
    for k in np.arange(0, seconds, beat):
        i = int(k * fs)
        n = int(0.15 * fs)
        env = np.exp(-np.arange(n) / (0.03 * fs))
        kick = np.sin(2 * np.pi * (50 + 80 * env) * np.arange(n) / fs) * env * 0.6
        out[i:i + n] += kick[: len(out) - i, None]
        j = int((k + beat / 2) * fs)
        m = int(0.05 * fs)
        if j + m < len(out):
            out[j:j + m] += (rng.standard_normal((m, 2)) * np.exp(-np.arange(m) / (0.01 * fs))[:, None]) * 0.15
    # bass line
    notes = [41.2, 41.2, 55.0, 49.0]
    f = np.array([notes[int(x / (2 * beat)) % 4] for x in t])
    out += (0.3 * np.sin(2 * np.pi * np.cumsum(f) / fs))[:, None]
    # "voice": vibrato sine with formant-ish harmonics, panned centre, silent first 8 s
    vib = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(vib) / fs
    voice = sum(np.sin(h * ph) / h for h in range(1, 8)) * 0.15 * (t > 8)
    out += voice[:, None]
    # pad, panned
    pad = 0.05 * (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392 * t))
    out[:, 0] += pad
    out[:, 1] += 0.5 * pad
    return (out / np.max(np.abs(out)) * 0.8).astype(np.float32)


def snr(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - est) ** 2), 1e-12))


def read_wav16(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data[44:], dtype=np.int16).reshape(-1, 2).astype(np.float32) / 32767


x48 = song()
raw = os.path.join(WORK, "raw.f32")
prep = os.path.join(WORK, "prepared.f32")
x48.tofile(raw)
run(CLI, "prepare", raw, "48000", "2", prep)
x = np.fromfile(prep, np.float32).reshape(-1, 2)
expected = int(len(x48) * 44100 / 48000)
assert abs(len(x) - expected) <= 1, (len(x), expected)

stats = last_json(run(CLI, "analyze", prep))
ref_lufs = pyloudnorm.Meter(44100).integrated_loudness(x.mean(axis=1).astype(np.float64))
print(f"LUFS engine {stats['lufs']:.2f}  pyloudnorm {ref_lufs:.2f}")
assert abs(stats["lufs"] - ref_lufs) < 0.5

results = {}
for label, threads, chunk in [("chunked", "3", "10"), ("single", "1", "1000")]:
    out = os.path.join(WORK, label)
    os.makedirs(out, exist_ok=True)
    rss_file = os.path.join(WORK, f"{label}.rss")
    res = last_json(run("/usr/bin/time", "-f", "%M", "-o", rss_file, CLI, "separate", MODEL, prep, out, threads, chunk))
    rss_mb = int(open(rss_file).read().split()[-1]) / 1024
    audio_s = len(x) / 44100
    print(f"{label}: {threads} thread(s), {res['seconds']} s for {audio_s:.0f} s of audio "
          f"({res['seconds'] / audio_s:.1f}x real time), peak RSS {rss_mb:.0f} MB")
    results[label] = {s: read_wav16(os.path.join(out, f"{s}.wav")) for s in SOURCES}
    for s in SOURCES:
        assert len(results[label][s]) == len(x), (s, len(results[label][s]), len(x))

total = sum(results["chunked"].values())
print(f"stems sum vs mix SNR: {snr(x, total):.1f} dB")
assert snr(x, total) > 10

# Two runs differ a little even unchunked (Demucs shifts each run by a random
# offset), so the difference is measured against the mix, not the stem: a
# quiet stem's own SNR says little about whether the chunk joins are clean.
mix_power = np.sum(x ** 2)
for s in SOURCES:
    a, b = results["single"][s], results["chunked"][s]
    level = 10 * np.log10(np.mean(a ** 2) + 1e-12)
    err = 10 * np.log10(mix_power / max(np.sum((a - b) ** 2), 1e-12))
    print(f"  {s:7s} chunked vs single: difference {err:.1f} dB below the mix (stem level {level:.0f} dB)")
    assert err > 20, s
print("engine OK")
