# Analysis stages

What runs after separation, on the phone: the beat grid for the click track
(and the BPM and tempo stability from it), the key, and the click renderer
for exports. Each is a C++ port of desktop StemDeck's own Python, checked
against it by [`scripts/analysis_parity.py`](../scripts/analysis_parity.py)
on every push (the `analysis` job in CI):

| Stage | Desktop StemDeck | Here |
|---|---|---|
| Decode to mono 22.05 kHz | `ffmpeg -ac 1 -ar 22050` | `src/swr.cpp` (same samples as ffmpeg 7, to float rounding) |
| Beat and downbeat detection | [Beat This!](https://github.com/CPJKU/beat_this) `final0` in PyTorch | the same model exported to ONNX (`scripts/export_models.py`), on ONNX Runtime; front end and post-processing in `src/beat_this.cpp` |
| Beat grid clean-up | `app/pipeline/beatgrid.py` | `src/beatgrid.cpp`, with librosa's onset strength, peak picking and onset detection in `src/librosa.cpp` |
| Key | `app/pipeline/analyze.py` (`refine_key_from_stems`) | `src/key.cpp`, with librosa's `chroma_cqt` and `estimate_tuning` in `src/cqt.cpp` |
| Click render | `app/pipeline/click_render.py` | `src/click.cpp`, pinned to the same fixture as desktop's (`test/fixtures/click_levels.json`) |

What the parity check measures (release notes have the latest numbers): beat
and downbeat counts and every beat time, the bar marks and every other field
of `beats.json`; the key, scale and confidence on desktop's labelled key set
(`tests/keyset.py`) and on recordings; the click levels against the fixture
and the rendered click sample for sample.

The one part that is not desktop's exactly is the resampler between the
constant-Q transform's octaves, which librosa delegates to libsoxr: here it
is a linear-phase FIR with soxr HQ's band edges. The chroma it gives agrees
with librosa's to about 1e-5.

ONNX Runtime is loaded at run time (`src/ort.cpp`): on the phone the library
from the `onnxruntime-android` package, on a computer the one from ONNX
Runtime's release archive (`SDA_ORT_LIB`). `third_party/onnxruntime` holds
its C API header (MIT), from the same version.

## Building and checking on a computer

```sh
cmake -S analysis -B build/analysis -DCMAKE_BUILD_TYPE=Release
cmake --build build/analysis
python3 scripts/analysis_parity.py --cli build/analysis/sdanalysis_cli \
    --upstream <stemdeck checkout> --beat-ckpt final0.ckpt \
    --beat-onnx beat_this_final0.onnx --ort-lib libonnxruntime.so --audio <files>
```

The reference side needs desktop StemDeck's Python dependencies
(`scripts/parity-requirements.txt`, CPU torch). The CI job shows the exact
steps.

## Credits

Ports of: StemDeck (Apache-2.0, © the StemDeck contributors); librosa
0.11 (ISC); numpy's reductions (BSD-3-Clause); Beat This! (MIT, © CPJKU).
