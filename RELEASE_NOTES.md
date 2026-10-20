StemDeck for Android 1.1: the song's beat grid, key and a click track, worked out on the phone the way desktop StemDeck does it, and every stem in one ZIP.

This is an unofficial port. It is not made or supported by the StemDeck project.

**Install:** tap `StemDeck-Android.apk` below on your phone, and allow your browser to install apps when Android asks. It installs over 1.0 and keeps your library. Requires Android 10 or newer on a 64-bit phone.

### New
- **Beat grid and tempo.** After separating, the app finds the beats and bar lines on the drums stem with Beat This!, the model desktop StemDeck uses, then cleans the grid up exactly as desktop does. The Mixer shows the beats under the waveform; Analysis shows the tempo and how many beats land on a drum hit. The first song downloads the beat model once (82 MB).
- **Key.** Found from the guitar, piano, other, vocals and bass stems with desktop's method (constant-Q chroma, its key profiles and its relative-key rules), with how sure it is.
- **Click track.** A metronome on the detected beats in the Mixer: on/off, volume, accent (auto from the bars, off or a fixed meter), half or double time, and a count-in. It stays in time at any speed. *Export mix* can include it.
- **Export stems (ZIP).** The stems that are on in the Mixer, in one ZIP, stored as they are.

### Same as desktop
Checked against desktop StemDeck's Python on every build: the same beats and bars on test songs (fast, slow and rubato), the same key on desktop's whole 33-song key set, and click renders that match sample for sample.

### Privacy
The app downloads its models (separation 55 MB, beat 82 MB) once each; nothing you import leaves the phone. The fonts are now part of the app, so opening it no longer contacts Google Fonts.

### Not yet on Android
YouTube/SoundCloud links, lyrics, the lead/backing vocal split and sections.

Songs separated with 1.0 have no beat grid or key; separate them again to get them.

Only use this app with audio you have the right to use.
