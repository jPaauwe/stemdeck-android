# StemDeck Bench

A separate Android app that measures how fast
[StemDeck for Android](../README.md)'s separation engine runs on your phone,
and where the time goes. It installs next to StemDeck and doesn't touch its
library.

**Download:** [`StemDeck-Bench.apk`](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Bench.apk)
from the newest release. To compare the engine *builds* on a song of your own,
use [StemDeck Compare](../compare/README.md) instead.

## Running it

1. Open StemDeck Bench. The first run downloads the model (about 55 MB).
2. Pick a test-song length and the thread counts to try (1, 2 and 4 are
   ticked by default).
3. Tap **Run benchmark** and leave the phone alone. The screen stays on. A
   full run takes several minutes, and longer on more thread counts.
4. Tap **Copy** or **Share** to send the results.

For fair numbers, start with a cool phone that isn't charging, and close other
apps.

## What it measures

The app contains two engines: the **optimized** engine from this build, and
a **baseline**, a frozen copy of the engine as it was before optimizing
started (commit `b76c67a`, in `src/main/cpp/baseline`).

1. **One 7.8-second segment on one thread**, the unit Demucs works in. Both
   engines run it twice and the best time is kept. Then:
   - how close the optimized output is to the baseline, per stem (SNR in dB;
     above 60 is identical to the ear),
   - where the time goes in the optimized engine, per stage (STFT, encoder,
     transformer, decoder…) and per operation (convolutions, attention,
     normalization…).
2. **A full synthetic test song** separated as the app does, at each thread
   count you picked: time, speed relative to the song's length, peak memory
   and the phone's thermal state. Optionally the baseline too.

The report starts with the phone model, chip, Android version, core count and
RAM, so results from different phones can be told apart.

## How it is built

- `src/main/java/org/stemdeckport/bench/`: the UI (`BenchActivity.kt`), the
  model download (`ModelManager.kt`) and the JNI bindings (`Natives.kt`).
- `src/main/cpp/`: `bench_jni.cpp` binds the shared benchmark code in
  `engine/bench/bench_core.cpp` twice, once to each engine. The optimized
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
