# StemDeck Bench

A separate Android app that measures how fast
[StemDeck for Android](../README.md)'s separation engine runs on your phone,
and where the time goes. It installs next to StemDeck and doesn't touch its
library.

**Download:** [`StemDeck-Bench.apk`](https://github.com/jPaauwe/stemdeck-android/releases)
from the newest development release (pre-release); it is not part of the public releases. To compare the engine *builds* on a song of your own,
use [StemDeck Compare](../compare/README.md) instead.

## Running it

1. Open StemDeck Bench. The first run downloads the model (about 55 MB).
2. Pick a test-song length and how many segments to separate at once (1, 2
   and 3 are ticked by default). Each runs CPU only and with the GPU.
3. Tap **Run benchmark** and leave the phone alone. The screen stays on. A
   full run takes several minutes, longer with more settings ticked.
4. Tap **Copy** or **Share** to send the results.

For fair numbers, start with a cool phone that isn't charging, and close other
apps.

## What it measures

The app contains three engines (see [`engines/`](../engines/README.md)):

- **original demucs.cpp**: upstream, unmodified (`engines/original`),
- **builds 4–10**: the first released StemDeck engine (`engines/b4`),
- **app engine**: the engine the app uses now (`/engine`, the one from
  build 24): build 13 using every core and the GPU.

The report starts with what the app's engine runs on, for example
`10 cores, GPU: Samsung Xclipse 940 (Vulkan 1.3)`, or why the GPU is not
used.

1. **One 7.8-second segment**, the unit Demucs works in: the original and
   builds 4–10 on one thread, and the app's engine on 1 core, on every core,
   and on every core plus the GPU. Each runs twice and the best time is
   kept. Then:
   - each one's speed relative to the original,
   - how close each output is to the original's, per stem (SNR in dB; above
     60 is identical to the ear),
   - where the time goes in the fastest setup, per stage (STFT, encoder,
     transformer, decoder…) and per operation (convolutions, attention,
     normalization, and `gpu: …` for work done on the GPU).
2. **A full synthetic test song** through the real separation path: the
   app's engine with each number of segments at once you picked, CPU only
   and with the GPU. Optionally the original (1 thread, the whole song at
   once) and builds 4–10 (2 threads) run too. You get the time, speed
   relative to the song's length, peak memory and the phone's thermal
   state.

3. **The analysis after separation** (app 1.1), on the test song's stems:
   the beat grid (Beat This! on ONNX Runtime, then desktop's clean-up) with
   the CPU provider at the default, 2, 4, 6 and 8 threads, XNNPACK and
   NNAPI, each with its time, the share spent in the model, peak memory and
   whether the beats match the first run; then the key from the stems. The
   first run downloads the beat model (82 MB).

The report starts with the phone model, chip, Android version, core count and
RAM, so results from different phones can be told apart.

## How it is built

- `src/main/java/org/stemdeckport/bench/`: the UI (`BenchActivity.kt`), the
  model download (`ModelManager.kt`) and the JNI bindings (`Natives.kt`).
- `src/main/cpp/`: `bench_jni.cpp` binds the shared benchmark code in
  `engine/bench/bench_core.cpp` once to each engine. The optimized
  engine is built with `SD_PROFILE`, which turns on the per-stage and
  per-operation timers (`engine/demucs/sdprof.hpp`). Both engines are built
  with hidden symbols, so they can't clash in one process.

The same benchmark runs on a computer and in CI, without the app:

```sh
cmake -S engine -B build/engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/engine --target sdbench_cli sdbench_cli_baseline
build/engine/sdbench_cli segment <model.bin> profile.json
```

`sdbench_cli` also has `song`, `separate` and `compare` subcommands.

## Licence

Same as StemDeck for Android: see the [main README](../README.md#credits-and-licences).
