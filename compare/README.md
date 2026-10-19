# StemDeck Compare

A separate Android app that runs one song through several builds of
[StemDeck for Android](../README.md)'s separation engine and compares them:
how fast each one is, how much memory it needs, and how different its stems
are. Use it to check that a speed-up didn't change the sound, or to hear what
did change.

**Download:** [`StemDeck-Compare.apk`](https://github.com/jPaauwe/stemdeck-android/releases/latest/download/StemDeck-Compare.apk)
from the newest release. It installs next to StemDeck and doesn't touch its
library.

## Running it

1. Tap **Choose MP3**, or share a song to StemDeck Compare from another app.
   Any audio format Android can decode works.
2. Tick the engine builds to compare. Builds that share an engine are one
   entry, for example "builds 4–10".
3. Pick the **reference** build that the others are compared with (it always
   runs first) and the number of **threads**.
4. Tap **Run** and leave the phone alone. The screen stays on, and every build
   separates the whole song, so this takes a while. The first run also
   downloads the model (about 55 MB).
5. Tap **Copy** or **Share** to send the results.

Options:

- **Let the phone cool down between builds** (on by default) waits at least 20
  seconds, and until Android reports no thermal pressure (at most 3 minutes),
  before each build. Without it, later builds run on a hotter, throttled CPU and
  look slower than they are.
- **Save every build's stems** writes them to
  `Download/StemDeck Compare/<song>/<build>/` so you can listen to them side by
  side.

## The report

- **Speed:** per build, the separation time, that time as a multiple of the
  song's length, speed-up against the reference, peak memory, and the thermal
  state before and after.
- **Difference from the reference:** per stem and overall, the
  signal-to-noise ratio of each build's stems against the reference's. Above
  60 dB is identical, 40–60 dB is inaudible, 20–40 dB is a small difference,
  below 20 dB is clearly different. For each build it names the stem and the
  second of the song where they differ most, so you know where to listen.
- **Difference from the previous build:** the same, between consecutive
  builds.
- **Stems add back up to the song:** how closely the six stems sum back to
  the original mix. Higher is better. It doesn't need a reference, so it's a
  rough quality check for each build on its own.
- **Stem levels:** each stem's loudness (RMS dBFS), to spot a build that
  moves sound from one stem to another.

The song is decoded and resampled once, with the reference build, so every
build gets exactly the same input.

## How it is built

The engine builds come from git: at build time,
`scripts/collect_engines.py` looks at every `build-*` tag, keeps one copy per
distinct engine source, and adds the working tree's engine when it differs
from the newest tag (listed as "this build"). New engine builds therefore
show up in the next APK without any changes here.

Each engine version is compiled into its own library,
`libsdengine_<id>.so` (`src/main/cpp/plugin.cpp` includes that version's
`engine.cpp`), which exports only a small C API (`sdp_load_model`,
`sdp_prepare`, `sdp_separate`, `sdp_cancel`…). `compare_jni.cpp` loads one
version at a time with `dlopen`, runs it, unloads it, and computes the
differences between the WAV files the versions write.

Building it needs Python 3 and the full git history with tags:

```sh
git fetch --tags --unshallow   # if your clone is shallow
./gradlew :compare:assembleRelease
```

## Licence

Same as StemDeck for Android: see the [main README](../README.md#credits-and-licences).
