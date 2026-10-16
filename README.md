# StemDeck for Android (unofficial port)

[StemDeck](https://github.com/stemdeckapp/stemdeck) on your phone, with the
separation running **on the phone itself**: import a song, get vocals, drums,
bass, guitar, piano and other on their own lanes, and practice with them.
No computer, no account, nothing uploaded.

This is a personal, unofficial port. It is not made or supported by the
StemDeck project.

## Install

1. On your phone, open the repository's **Releases** page and tap
   `StemDeck-Android.apk` on the newest build.
2. Allow your browser to install apps when Android asks (one time).
3. Open StemDeck. The first separation downloads the model (about 55 MB).

Each push to `main` builds and publishes a new release. With the signing key
in the repository secrets (see `signing/README.md`), every build is signed with
the same key, so a new APK installs over the old one and keeps your library.

## Using it

- **Extract** → *Upload audio file* → pick stems → *Extract stems*. Or, from
  the Files app or a download, **Share → StemDeck**.
- Separation runs in the background with a notification (tap *Cancel* there to
  stop). The screen can be off.
- **Mixer**: mute, solo and volume per stem, speed and key, seek. **Analysis**:
  loudness, peak, dynamic range and how present each stem is.
  *Export mix* saves what you hear to `Download/StemDeck/`.
- Stems are 16-bit WAV files in
  `Android/data/org.stemdeckport.android/files/jobs/<id>/stems/`, reachable
  over USB from a computer. They are removed if you uninstall the app.

## How it works

| Desktop StemDeck | This app |
|---|---|
| Python/FastAPI server | Kotlin server on `127.0.0.1` inside the app (`LocalServer.kt`), same API shape for the parts the mobile UI uses |
| PyTorch Demucs `htdemucs_6s` | [demucs.cpp](https://github.com/sevagh/demucs.cpp), a C++ implementation of the same model and weights, compiled for arm64 (`app/src/main/cpp`) |
| FFmpeg decode | Android's built-in decoders (`AudioDecoder.kt`) and a windowed-sinc resampler to 44.1 kHz |
| Mobile web UI (`static/mobile`) | The same UI, bundled in `app/src/main/assets/web` and shown in a WebView. Changes are marked `Android:` |
| pyloudnorm | The same BS.1770 K-weighting and gating in C++ |

The song is separated in ~24-second chunks, a few in parallel, and blended
back together over a one-second overlap. Normalization uses the whole song's
statistics, as PyTorch Demucs does, so chunking does not change the result.
CI checks this against a one-pass separation with the real model on every
push (`scripts/engine_test.py`).

## Not yet on Android

Planned for later stages: key and BPM detection, click track and beat grid,
YouTube/SoundCloud links, lyrics, the lead/backing vocal split, sections,
exporting every stem as a zip.

## Building yourself

```sh
git clone --recurse-submodules <this repo>
./gradlew assembleRelease
```

Needs the Android SDK with NDK 27.2.12479018 and CMake 3.22.1 (Android Studio
installs them on first sync).

## Credits and licences

- StemDeck, the mobile UI and audio engine in `app/src/main/assets/web`:
  Apache License 2.0, © the StemDeck contributors. See
  `app/src/main/assets/web/LICENSE-StemDeck.txt`; the upstream commit is in
  `UPSTREAM_COMMIT` next to it.
- demucs.cpp: MIT, © Sevag H (`app/src/main/cpp/demucs/LICENSE`). One
  addition, `demucs_inference_normalized`, is marked in `model_apply.cpp`.
- Demucs and the `htdemucs_6s` weights: MIT, © Meta Platforms.
- Eigen: MPL 2.0. NanoHTTPD: BSD 3-Clause.
- The SoundTouch-based and Signalsmith Stretch worklets in
  `assets/web/vendor`, as shipped by StemDeck, under the MIT licences in
  their headers.

This app is for audio you have the right to use.
