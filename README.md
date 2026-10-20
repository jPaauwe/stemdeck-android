# StemDeck for Android (unofficial port)

[StemDeck](https://github.com/stemdeckapp/stemdeck) on your phone, with the
separation running **on the phone itself**: import a song, get vocals, drums,
bass, guitar, piano and other on their own lanes, and practice with them,
with the song's beat grid, key and a click track.
No computer, no account, nothing uploaded (see [Privacy](#privacy)).

This is a personal, unofficial port. It is not made or supported by the
StemDeck project.

## Install

1. On your phone, download
   [`StemDeck-Android.apk`](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Android.apk)
   from the latest release (or open the repository's **Releases** page and
   tap it there). Requires Android 10 or newer on a 64-bit (arm64) phone.
2. Allow your browser to install apps when Android asks (one time).
3. Open StemDeck. The first separation downloads the model (about 55 MB):
   Demucs' `htdemucs_6s` weights in demucs.cpp's format (bit for bit the
   weights Demucs v4.1.0 publishes; CI checks this).

Public releases are `v1.0.0`, `v1.0.1`, …: the version in `VERSION`. The
first push to `main` after `VERSION` changes builds and publishes that
release (the app only, with `RELEASE_NOTES.md`); other pushes publish
development builds as pre-releases, with the two benchmark apps. With the signing key
in the repository secrets (see `signing/README.md`), every build is signed with
the same key, so a new APK installs over the old one and keeps your library.

## Using it

- **Extract** → *Upload audio file* → pick stems and the file format →
  *Extract stems*. Or, from the Files app or a download, **Share → StemDeck**.
- Separation runs in the background with a notification (tap *Cancel* there to
  stop). The screen can be off.
- **Mixer**: mute, solo and volume per stem, speed and key, seek. Under the
  waveform, the song's beats and bar lines. **Click**: a metronome on the
  detected beats, in time with the song at any speed, with volume, accent
  (auto from the detected bars, off, or a fixed meter), half or double time
  and a count-in of 1 to 4 bars.
- **Analysis**: loudness, peak, dynamic range, how present each stem is, the
  key (with how sure it is), the tempo and the beat grid.
  *Export mix* saves what you hear to `Download/StemDeck/`, with the click
  track if you tick *Include the click track*. *Export stems (ZIP)* saves the
  stems that are on in the Mixer, as they are stored, in one ZIP.
- After separating, the app finds the beat grid on the drums stem and the
  key from the guitar, piano, other, vocals and bass stems, the way desktop
  StemDeck does. The first song downloads the beat model once (82 MB). If a
  song has no grid (too few beats, say) the Analysis tab says why.
- Stems are **320 kbps MP3** files by default (about 2.4 MB a minute each).
  Choose **WAV** under *File format* on the Extract screen for lossless 16-bit
  stems (about 10 MB a minute each). The choice is remembered, applies to new
  songs and to *Export mix*, and songs already in the library keep their
  format. The MP3s are gapless, so the stems stay in sync with each other and
  with the original.
- Stems are in
  `Android/data/org.stemdeckport.android/files/jobs/<id>/stems/`, reachable
  over USB from a computer. They are removed if you uninstall the app.

## How it works

| Desktop StemDeck | This app |
|---|---|
| Python/FastAPI server | Kotlin server on `127.0.0.1` inside the app (`LocalServer.kt`), same API shape for the parts the mobile UI uses |
| PyTorch Demucs `htdemucs_6s` | [demucs.cpp](https://github.com/sevagh/demucs.cpp), a C++ implementation of the same model and weights, compiled for arm64 and optimized for phones (`engine/`) |
| FFmpeg decode | Android's built-in decoders (`AudioDecoder.kt`) and a windowed-sinc resampler to 44.1 kHz |
| Mobile web UI (`static/mobile`) | The same UI, bundled in `app/src/main/assets/web` and shown in a WebView. Changes are marked `Android:` |
| pyloudnorm | The same BS.1770 K-weighting and gating in C++ |
| Beat grid: Beat This! (PyTorch) + `beatgrid.py` (librosa, numpy) | Beat This! exported to ONNX and run with ONNX Runtime; its log-mel front end, ffmpeg's downmix and resampler, and the clean-up ported to C++ (`analysis/`) |
| Key: `analyze.py` (librosa `chroma_cqt`) | The constant-Q chroma, tuning estimate and key logic ported to C++ (`analysis/`) |
| Click track: `metronome.js` / `click_render.py` | `metronome.js` unchanged in the WebView; the renderer ported to C++ for *Export mix* |

The engine separates the song the way PyTorch Demucs does: normalized with
the whole song's statistics, cut into Demucs' own 7.8-second segments with a
25% overlap and a time shift, and blended back with triangular weights. The
segments are spread over a few worker threads (1 to 3, depending on the
phone's memory), so memory stays flat however long the song is.

The analysis stages are checked against desktop StemDeck's own Python on
every push (`scripts/analysis_parity.py` in CI): the beat grid gives the same
beats and bars on synthetic bands, click tracks and recordings (fast and
rubato ones included); the key agrees on all 33 songs of desktop's labelled
key set and on the recordings; the click renders are sample for sample the
same, and the JavaScript scheduler matches desktop's shared fixture. The
details are in [`analysis/README.md`](analysis/README.md).

## Privacy

Separation and analysis happen on the phone; nothing you import is uploaded,
and there is no account. The app goes online only to download its model
files, each once, the first time a feature needs it, and checks each against
a pinned SHA-256:

- the separation model (55 MB) from Hugging Face
  (`huggingface.co/datasets/Retrobear/demucs.cpp`),
- the beat model (82 MB) from this repository's `models-1` release on GitHub.

Those requests carry nothing about you or your songs. The interface's fonts
are bundled with the app, so opening it contacts nobody. ONNX Runtime's
telemetry is switched off.

## Speed

Separation is the slow part. The app's engine is the one from development
build 24 (`engines/b13plus`): Demucs' own segments over the whole song,
faster attention, convolutions and normalization, convolutions that need a
third of the memory, and **every core and the GPU**:

- **Every core:** the segments that fit in memory run side by side, and the
  heavy maths inside each one (matrix products, attention, norms,
  activations) is split into small pieces that any idle core picks up, so
  the fast and slow cores of a phone chip each do what they can.
- **GPU:** the transformer (projections, attention, feed-forward) and the
  large convolution products run on the GPU through Vulkan
  (`engine/compute/`). The GPU checks itself against the CPU on first use;
  when it is missing, fails, or is busy with another segment, that work runs
  on the CPU instead. *Processing* on the Extract screen switches the GPU
  off (*CPU only*), and the progress line shows what a song is separated on.

The results are the same as the original demucs.cpp's, to float rounding,
given the same time shift. On a 2-core test
computer build 13 separated a song about 4x faster than the original
demucs.cpp (on 2 threads; the original uses 1), with the same output when
given the same time shift.
Every earlier engine is kept in [`engines/`](engines/README.md), with the
measurements. The newest engine, with Demucs v4.1.0's weights, is also
published on its own as a library for other apps:
[demucs-android](https://github.com/jPaauwe/demucs-android). CI checks the engine on every push (`scripts/engine_test.py`
and the host benchmark in `engine/test/bench_cli.cpp`): separation with the
real model, and the app's engine against the first released one.

Two separate apps measure the engines on the phone itself. Each release has
both:

- **[StemDeck Bench](bench/README.md)**
  (in the development releases) profiles the
  engines: the original demucs.cpp, builds 4–10 and the app's engine, where
  the time goes per stage and per operation, and how speed and memory scale
  with threads.
- **[StemDeck Compare](compare/README.md)**
  (in the development releases) runs
  your own song through the original demucs.cpp, builds 4–10, build 13 and
  the app's engine,
  compares their speed, memory and stems, and lets you listen to them side
  by side.

## Not yet on Android

Planned for later stages: YouTube/SoundCloud links, lyrics, the
lead/backing vocal split, sections.

## Building yourself

```sh
git clone --recurse-submodules <this repo>
./gradlew assembleRelease
```

This builds all three apps: `app/` (StemDeck), `bench/` and `compare/`. The
app's separation engine is in `engine/` (the benchmark apps also use the
kept engines in `engines/`), and it builds on a computer too:

```sh
cmake -S engine -B build/engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/engine --target sdengine_cli sdbench_cli
```

Needs the Android SDK with NDK 27.2.12479018 and CMake 3.22.1 (Android Studio
installs them on first sync). StemDeck Compare also needs Python 3.

## Credits and licences

- StemDeck, the mobile UI and audio engine in `app/src/main/assets/web`:
  Apache License 2.0, © the StemDeck contributors. See
  `app/src/main/assets/web/LICENSE-StemDeck.txt`; the upstream commit is in
  `UPSTREAM_COMMIT` next to it.
- demucs.cpp: MIT, © Sevag H (`engine/demucs/LICENSE`, upstream commit in
  `UPSTREAM_COMMIT`). Changed here for memory and speed: the convolutions
  (`conv.hpp`), normalization, GLU, layer scale and attention
  (`layers.cpp`) and `demucs_inference_normalized` (`model_apply.cpp`).
  Results match the original to float rounding. Each change is marked
  "StemDeck Android". The unmodified original is in `engines/original`.
- Demucs ([adefossez/demucs](https://github.com/adefossez/demucs)) and the
  `htdemucs_6s` weights: MIT, © Meta Platforms.
- Eigen: MPL 2.0. NanoHTTPD: BSD 3-Clause.
- [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, Johannes Kepler
  University Linz), code and the `final0` weights: MIT. The app downloads an
  ONNX export of the published checkpoint, made reproducibly by
  `scripts/export_models.py` (`models/models.json`).
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.31.0: MIT
  (`analysis/third_party/onnxruntime`).
- The analysis ports in `analysis/` follow desktop StemDeck's Python
  (Apache 2.0) and the parts of [librosa](https://librosa.org) (ISC),
  numpy (BSD), torchaudio (BSD) and FFmpeg's resampler they call; the code is
  written for this app, and FFmpeg's is not used.
- Space Grotesk and JetBrains Mono (the interface's fonts): SIL Open Font
  License 1.1 (`assets/web/fonts`).
- [LAME](https://lame.sourceforge.io) 3.100, the MP3 encoder and decoder:
  LGPL 2.0 or later. It is not stored in this repository: the build downloads
  the release tarball, checks its SHA1 (`app/src/main/cpp/lame/CMakeLists.txt`)
  and builds it unchanged as its own library, `libmp3lame.so`, which the app
  links dynamically.
- The SoundTouch-based and Signalsmith Stretch worklets in
  `assets/web/vendor`, as shipped by StemDeck, under the MIT licences in
  their headers.

This app is for audio you have the right to use.
