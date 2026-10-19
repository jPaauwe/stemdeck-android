# Engine archive

Every separation engine StemDeck for Android has had, each as a complete
source folder, so any of them can be built again later. `engines.txt` lists
them; `SOURCE.txt` in each folder says where it came from.

| Folder | Builds | What it is | In use |
|---|---|---|---|
| `original` | none | Upstream [demucs.cpp](https://github.com/sevagh/demucs.cpp) (commit f1206e9), unmodified, with a small wrapper. It runs the whole song through `demucs_inference` in one call on one thread. | Compare, Bench |
| `stage1` | 1–3 (never released) | The first port: upstream plus whole-song normalization, the song split into 24 s chunks. About 1.7 GB per thread. | archived |
| `b4` | 4–10 | The first released engine: as stage 1, with the transposed convolutions rewritten to need about 0.6 GB per thread. | Compare, Bench |
| `b12` | 12 | Faster norms, GLU, layer scale, attention and stride-1 convolutions. Same output. | archived |
| `b13` | 13, 20 | Demucs' own 7.8 s segments over the whole song with a time shift, as PyTorch Demucs does, plus build 12's speed-ups. | Compare |
| `b14` | 14–19 | Build 13 plus vectorized GELU and blocked layout transposes. Same output. | archived |
| `b13plus` | 22 on | Build 13 using **every core and the GPU**: a shared thread pool splits the maths inside each segment over all cores, and Vulkan compute shaders run the transformer and the large convolution products on the GPU. Same output as build 13. | **the app**, Compare, Bench |

The app's engine is in `/engine` and is identical to `b13plus`.

## How they compare

Measured on a 20-second test song on a 2-core x86 computer (all six stems):

| Engine | Time | Peak memory | Output |
|---|---|---|---|
| original demucs.cpp | 267 s (1 thread) | 1.9 GB | the reference |
| builds 4–10 | 204 s (2 threads) | 0.9 GB | identical (67–102 dB) |
| build 13 | 63 s (2 threads) | 1.5 GB | 7–30 dB from the original |
| build 13 with the original's shift | 58 s (2 threads) | 1.3 GB | identical (57–94 dB) |

Build 13 does exactly the same maths as the original. The difference comes
only from the **time shift**. Demucs moves the song by a random amount, up to
0.5 s, before cutting it into segments. Upstream picks that amount with
`rand()`, and build 13 picks it from the song's length. A different shift
moves every segment boundary, which changes the stems by roughly 15–30 dB,
and more on near-silent stems. That is the same variation PyTorch Demucs
shows between two runs. With the same shift, build 13 matches the original.

Builds 4–10 match the original on this short song because it fits in one
24 s chunk, so their second thread had nothing to do. On longer songs they cut the song into 24 s chunks, each with its
own shift, so they differ from the original there.

## Using an archived engine again

- **StemDeck Compare:** set the last field of its line in `engines.txt` to
  `yes`, and the next build bundles it.
- **The app:** copy its `engine.cpp`, `engine.hpp` and `demucs/` over
  `/engine` (and `sdprof.cpp`, if it has one).
- **A new engine:** `tools/extract_from_git.py <repo> <out>` exports every
  distinct engine from the `build-*` tags. Add a new folder and a line to
  `engines.txt`.

Licences: demucs.cpp is MIT (`*/demucs/LICENSE`). The rest is this
repository's code.
