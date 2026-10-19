# StemDeck for Android (unofficial port)

[StemDeck](https://github.com/stemdeckapp/stemdeck) on your phone, with the
separation running **on the phone itself**: import a song, get vocals, drums,
bass, guitar, piano and other on their own lanes, and practice with them.
No computer, no account, nothing uploaded.

This is a personal, unofficial port. It is not made or supported by the
StemDeck project.

## Install

1. On your phone, download
   [`StemDeck-Android.apk`](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Android.apk)
   from the newest release (or open the repository's **Releases** page and tap
   it there).
2. Allow your browser to install apps when Android asks (one time).
3. Open StemDeck. The first separation downloads the model (about 55 MB).

Each push to `main` builds and publishes a new release. With the signing key
in the repository secrets (see `signing/README.md`), every build is signed with
the same key, so a new APK installs over the old one and keeps your library.

## Using it

- **Extract** → *Upload audio file* → pick stems and the file format →
  *Extract stems*. Or, from the Files app or a download, **Share → StemDeck**.
- Separation runs in the background with a notification (tap *Cancel* there to
  stop). The screen can be off.
- **Mixer**: mute, solo and volume per stem, speed and key, seek. **Analysis**:
  loudness, peak, dynamic range and how present each stem is.
  *Export mix* saves what you hear to `Download/StemDeck/`.
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

The engine separates the song the way PyTorch Demucs does: normalized with
the whole song's statistics, cut into Demucs' own 7.8-second segments with a
25% overlap and a time shift, and blended back with triangular weights. The
segments are spread over a few worker threads (1 to 3, depending on the
phone's memory), so memory stays flat however long the song is.

## Speed

Separation is the slow part. The app uses the **build 13** engine: Demucs'
own segments over the whole song, with faster attention, convolutions and
normalization. On a 2-core test computer it separated a song about 4x faster
than the original demucs.cpp (on 2 threads; the original uses 1), with the
same output when given the same time shift.
Every earlier engine is kept in [`engines/`](engines/README.md), with the
measurements. CI checks the engine on every push (`scripts/engine_test.py`
and the host benchmark in `engine/test/bench_cli.cpp`): separation with the
real model, and the app's engine against the first released one.

Two separate apps measure the engines on the phone itself. Each release has
both:

- **[StemDeck Bench](bench/README.md)**
  ([download the latest](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Bench.apk)) profiles the
  engines: the original demucs.cpp, builds 4–10 and the app's engine, where
  the time goes per stage and per operation, and how speed and memory scale
  with threads.
- **[StemDeck Compare](compare/README.md)**
  ([download the latest](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Compare.apk)) runs
  your own song through the original demucs.cpp, builds 4–10 and build 13,
  compares their speed, memory and stems, and lets you listen to them side
  by side.

## Not yet on Android

Planned for later stages: key and BPM detection, click track and beat grid,
YouTube/SoundCloud links, lyrics, the lead/backing vocal split, sections,
exporting every stem as a zip.

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
- Demucs and the `htdemucs_6s` weights: MIT, © Meta Platforms.
- Eigen: MPL 2.0. NanoHTTPD: BSD 3-Clause.
- [LAME](https://lame.sourceforge.io) 3.100, the MP3 encoder and decoder:
  LGPL 2.0 or later. It is not stored in this repository: the build downloads
  the release tarball, checks its SHA1 (`app/src/main/cpp/lame/CMakeLists.txt`)
  and builds it unchanged as its own library, `libmp3lame.so`, which the app
  links dynamically.
- The SoundTouch-based and Signalsmith Stretch worklets in
  `assets/web/vendor`, as shipped by StemDeck, under the MIT licences in
  their headers.

This app is for audio you have the right to use.
