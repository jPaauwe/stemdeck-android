StemDeck for Android: [StemDeck](https://github.com/stemdeckapp/stemdeck) on your phone, with the stem separation running **on the phone itself**. Nothing is uploaded, and there's no account.

This is an unofficial port. It is not made or supported by the StemDeck project.

**Install:** tap `StemDeck-Android.apk` below on your phone, and allow your browser to install apps when Android asks. Requires Android 10 or newer on a 64-bit phone. The first separation downloads the separation model once (about 55 MB).

### What it does
- **Separates a song into six stems**: vocals, drums, bass, guitar, piano and other. It uses Demucs' `htdemucs_6s` model, the same one desktop StemDeck uses. Import a file on the Extract screen, or share one to StemDeck from another app.
- **Separates on the phone**, in the background, with a notification you can cancel from. The engine uses every core and, where the phone supports it, the GPU (Vulkan), falling back to the CPU on its own. You can choose *CPU only* under *Processing*.
- **Saves stems as 320 kbps MP3** by default (gapless, so the stems stay in sync), or as lossless WAV under *File format*.
- **Mixer:** mute, solo and volume per stem, plus speed, key and seeking. **Analysis:** loudness, peak, dynamic range and how present each stem is. **Export mix** saves what you hear to `Download/StemDeck`.

### Fixed since the development builds
- The *Extract stems* button is no longer covered by the mini player while a track is playing.

### Not yet on Android
Key and BPM detection, click track and beat grid, YouTube/SoundCloud links, lyrics, the lead/backing vocal split, sections, and exporting every stem as a zip.

Only use this app with audio you have the right to use.
