# Handoff: StemDeck for Android, missing features (brief: `stemdeck-android-missing-features-prompt.md`)

Written 2026-10-20 00:30 (Amsterdam). Work stopped at Jeroen's request.

## Before you start: the repository is locked

At 00:16 Jeroen added a ruleset, **IdkSomeRules** (id 25437607). It covers every branch (`~ALL`) and blocks branch creation, updates, deletion and non-fast-forward pushes, with no bypass. Pushes are refused with "repository rule violations". Do not try to get around it. Ask Jeroen to lift it or to scope it.

It also breaks CI. Every job ends with a "Publish logs to a ci-logs branch" step that pushes to `ci-logs/<job>`, and that push is now refused. So jobs fail even when their real work passed. In run 39333488265 the engine job passed every test and failed only on that step. Fix: exclude `ci-logs/**` from the ruleset, or give those steps `continue-on-error: true`.

## Where the code is

| What | Where |
|---|---|
| Phase 1 code | branch `phase1-analysis` at `3c8bdd8` (on GitHub), draft PR #1 |
| Fixes after that, not pushed | local commit `ea7d9d6` on top of `3c8bdd8`, plus this file. Both are in `stemdeck-phase1.bundle` (sent with this handoff) |
| `main` | `5e0f7f4`, v1.0.0, untouched |

Restore the unpushed work: `git fetch stemdeck-phase1.bundle phase1-analysis && git checkout FETCH_HEAD`. If you get "refusing to fetch into branch", use `git bundle unbundle`.

`ea7d9d6` contains:
- `ModelStoreTest` now uses a plain `ServerSocket` server. The JVM unit tests compile against android.jar, which has no `com.sun.net.httpserver`, and that broke the APK job. Both tests pass locally.
- `ort.cpp` now checks the statuses of the shape getters.
- `key.cpp` no longer captures structured bindings (an NDK warning).

## Phase 1 (1.1.0) status: code done, CI not green

**Built** (details in PR #1's description and the `3c8bdd8` commit message):
- Beat grid: Beat This! final0 exported to ONNX, run by ONNX Runtime 1.31.0 from C++.
- Key detection: a port of `analyze.py`, CQT chroma.
- Click playback: `metronome.js`, copied verbatim from desktop.
- Click export: a port of `click_render.py` through the mixdown `click*` params.
- Stems ZIP: `/stems/all.zip`.
- `ModelStore`: one model registry with resume, size check and SHA-256 check.
- Job fields: `bpm`, `key`, `scale`, `key_confidence`, `tempo_stability`, `analysis_notes`, `stage_timings`.
- Post-separation stages in order: beat grid → key → MP3. Each is best-effort and records why it was skipped.

All analysis code is in `analysis/` (static lib `sdanalysis`, host CLI `sdanalysis_cli`) and the JNI is in `app/src/main/cpp/analysis_jni.cpp`.

**Parity, measured locally against desktop StemDeck at `UPSTREAM_COMMIT`:**
- Beat grid identical on 18 cases.
- Key: 33/33 on the keyset, plus 10 recordings with the same confidence.
- Click renders sample-identical, and the `click_levels.json` fixture passes.
- JS click parity 104/104.
- Playwright UI checks pass at 384×640, 740 and 832.

**CI on PR #1 (run 39333488265, Models run 39333471084):**
1. **APK**: failed in `:app:testReleaseUnitTest` (the `com.sun` import). Fixed in `ea7d9d6`, not pushed. The main Kotlin and C++ compiled for arm64 and x86_64. The 16 KB alignment and APK size check (`scripts/check_apk.py`) has not run yet.
2. **Engine**: real tests passed. It failed only on the ci-logs push (the ruleset).
3. **Analysis**: still running when work stopped. Its ci-logs push will fail for the same reason.
4. **Models**: failed in "Export and check". The log can't be read from a session (log download returns 403). The same script with the same pinned versions reproduces `80d04915…` byte for byte locally (12 s), so the CI export most likely differs in bytes, probably CPU-dependent constant folding. Next steps:
   - First add log capture to `models.yml` (tee to `models.log`, push to `ci-logs/models` like build.yml does) and read the real error.
   - If it is a SHA mismatch, decide: publish the CI export and update `models/models.json` to its SHA, or check numerically instead of bytewise (ONNX vs PyTorch < 1e-4 already passes).
   - `models-1` must exist with `beat_this_final0.onnx` matching the manifest before the app's download works. `Models.BEAT_THIS` in `ModelStore.kt` pins the SHA and the size (82,001,399 bytes).

**Left to do for 1.1.0, once CI is green:**
- README: remove beat grid, key, click and ZIP from "Not yet on Android".
- README: list the downloads (55 MB Demucs from Hugging Face, 82 MB beat model from the `models-1` GitHub release).
- README: make the privacy wording exactly true, including the Google Fonts load.
- README: credits for Beat This! (MIT), ONNX Runtime (MIT), librosa (ISC).
- `RELEASE_NOTES.md` 1.1.0, with the decisions: CQT chroma kept; models hosted on a `models-N` pre-release.
- `bench/README.md`: describe section 3, the analysis Bench.
- `VERSION` → 1.1.0, then merge to `main` with no CI run in progress. CI releases `v1.1.0`.
- Not measured on a phone yet: Jeroen runs StemDeck-Bench section 3 (beat grid and key timings, memory).

## Phases 2–4: not started in code

**Phase 2 research done (notes for whoever continues):**
- Desktop sections: `section_worker.py` + `section_refine.py` (novelty, beat snap, neutral "part" label, regularize). The All-In-One harmonix ensemble averages logits over 8 folds.
- Front end to port: madmom-infer `FramedSignal`. Frame 2048, fps 100 → hop 441.0. `origin` 0 means a centred frame: `start = int(i*hop) - 1024`. Out-of-range samples are padded with 0 (`pad=0`). Frame count with `end="normal"` is `ceil(len/hop)`. `remix` to mono is `np.mean(axis=-1)` in the signal's dtype.
- Then STFT → `FilteredSpectrogram` (12 bands/oct, fmin 30, fmax 17000, `norm_filters`) → `Log(mul=1, add=1)`. Stems in the order bass, drums, other, vocals; "other" is guitar + piano + other summed.
- lightcut ONNX folds take input `features` [1, 4, time, 81]. Outputs: `beat`, `downbeat`, `section`, `function` [*, 10, time] and `embedding` [1, *, time, 24]. Not yet checked against all-in-one-infer.
- **Licence flag for Jeroen:** the All-In-One HF card says MIT, but all-in-one-infer's `checkpoints.toml` lists CC-BY-NC-SA-4.0 for the weights. Settle this before shipping sections.
- Vocal split (UVR-MDX-NET Karaoke 2, MIT + credit) not started.

Phase 3 (LRCLIB lyrics, Settings → Network, off by default) and Phase 4 (link import spike) are not started. Follow the brief.

## Working notes for the next session

- Run `gh` from inside the repo, or it targets the wrong repo. GraphQL is blocked, so use `gh api repos/...` (REST). Actions logs and artifacts can't be downloaded (403). Read logs from the `ci-logs/<job>` branches instead, which need the ruleset fix above.
- pip, Hugging Face and Maven are blocked from the session container. The `dev-cache` pre-release (built by `dev/cache`'s `devcache.yml`) holds `wheels.tar.part00` (Python 3.13 CPU torch 2.6 env = `scripts/parity-requirements.txt`), `models.tar` (Beat This! checkpoints, lightcut / allinone / songbird files) and `audio.tar`. Delete that release and the `dev/cache` branch once they are no longer needed.
- Host build: `cmake -S analysis -B build/analysis -G Ninja && cmake --build build/analysis`. ORT on the host is loaded via `SDA_ORT_LIB=/path/libonnxruntime.so`. Parity: `scripts/analysis_parity.py --stages beatgrid,key,click` (needs the upstream StemDeck checkout at `UPSTREAM_COMMIT`).
- JVM unit tests without Gradle: compile with the Kotlin compiler jars in `/opt/gradle-8.14.3/lib`, against small stubs for `org.json`, `org.junit` and `android.util.Log`.
- Standing rules from the brief:
  - Don't push while a CI run is in progress.
  - Never touch `signing/` or `engines/`.
  - Bump `VERSION` only when a phase is finished and CI is green.
  - Commits end with the Co-Authored-By / Claude-Session lines.
  - Build only what is listed and report briefly.
