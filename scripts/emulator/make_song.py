#!/usr/bin/env python3
"""Writes a 3-minute 44.1 kHz stereo test song (drums, bass, a voice-like
lead and a pad) as a 16-bit WAV."""
import sys
import wave

import numpy as np

fs, seconds = 44100, 180
rng = np.random.default_rng(3)
t = np.arange(fs * seconds) / fs
out = np.zeros((len(t), 2))
beat = 0.5
for k in np.arange(0, seconds, beat):
    i, n = int(k * fs), int(0.15 * fs)
    env = np.exp(-np.arange(n) / (0.03 * fs))
    kick = np.sin(2 * np.pi * (50 + 80 * env) * np.arange(n) / fs) * env * 0.6
    out[i:i + n] += kick[: len(out) - i, None]
    j, m = int((k + beat / 2) * fs), int(0.05 * fs)
    if j + m < len(out):
        out[j:j + m] += rng.standard_normal((m, 2)) * np.exp(-np.arange(m) / (0.01 * fs))[:, None] * 0.15
notes = [41.2, 41.2, 55.0, 49.0]
f = np.array([notes[int(x / (2 * beat)) % 4] for x in t])
out += (0.3 * np.sin(2 * np.pi * np.cumsum(f) / fs))[:, None]
vib = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
ph = 2 * np.pi * np.cumsum(vib) / fs
out += (sum(np.sin(h * ph) / h for h in range(1, 8)) * 0.15 * (t > 8))[:, None]
pad = 0.05 * (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392 * t))
out[:, 0] += pad
out[:, 1] += 0.5 * pad
pcm = (out / np.max(np.abs(out)) * 0.8 * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(fs)
    w.writeframes(pcm.tobytes())
