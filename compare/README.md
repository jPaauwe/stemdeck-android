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
2. Tick the engines to compare: the **original demucs.cpp** (upstream,
   unmodified), **builds 4–10** (the first released StemDeck engine),
   **build 13**, and **v4.1.0 + all optimizations** (the app's engine).
   See [`engines/`](../engines/README.md).
3. Pick the **reference** that the others are compared with (it always runs
   first; the original by default) and the number of **threads**. The
   original always uses 1 thread and holds the whole song in memory, so it
   is the slowest and needs the most memory.
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
  Demucs shifts the song by a small random amount before separating, and
  each engine picks its own shift, so engines with identical maths still
  differ by roughly 15–30 dB (more on near-silent stems). See
  [`engines/README.md`](../engines/README.md#how-they-compare).
- **Difference from the previous build:** the same, between consecutive
  builds.
- **Stems add back up to the song:** how closely the six stems sum back to
  the original mix. Higher is better. It doesn't need a reference, so it's a
  rough quality check for each build on its own.
- **Stem levels:** each stem's loudness (RMS dBFS), to spot a build that
  moves sound from one stem to another.

The song is decoded and resampled once, with the reference build, so every
build gets exactly the same input.

## Listening

After a run, tap **Listen** to hear what the builds produced. The last run is
kept, so you can come back to it later, until you start a new one.

- **Stem:** drums, bass, other, vocals, guitar, piano, all stems together, or
  the original song.
- **Build:** tap another build while it plays. It switches at the same moment
  in the song, with a short crossfade, so you compare the same notes. Each
  build shows how far its stem is from the reference (SNR in dB).
- **Hear only the difference:** plays this build minus the reference, so you
  hear exactly what changed. Silence means they're the same. *Boost* raises it
  by 12 dB for quiet differences.
- **Where they differ most:** for each build, the second where its stem is
  furthest from the reference. Tap it to jump there and loop those 4 seconds.
- **Loop 5 s**, **−5 s** and **+5 s**, and a seek bar.
- **Blind test:** hides and shuffles the build names (A, B, C…). Pick the one
  you prefer, then tap **Reveal**.

## How it is built

The engines come from the archive in [`engines/`](../engines/README.md): at
build time, `scripts/engines_for_compare.py` takes the ones marked `yes` in
`engines/engines.txt`. When the app's engine (`/engine`) is not one of them,
it is added as "this build". To bring an archived engine back, mark it
`yes`.

Each engine version is compiled into its own library,
`libsdengine_<id>.so` (`src/main/cpp/plugin.cpp` includes that version's
`engine.cpp`), which exports only a small C API (`sdp_load_model`,
`sdp_prepare`, `sdp_separate`, `sdp_cancel`…). `compare_jni.cpp` loads one
version at a time with `dlopen`, runs it, unloads it, and computes the
differences between the WAV files the versions write. The Listen screen
(`ListenActivity.kt`) plays those WAV files through `StemPlayer.kt`, an
AudioTrack player that can switch source mid-stream.

Building it needs Python 3:

```sh
./gradlew :compare:assembleRelease
```

## Licence

Same as StemDeck for Android: see the [main README](../README.md#credits-and-licences).
