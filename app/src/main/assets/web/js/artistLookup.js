// Android: only artistNameKey from desktop's static/js/artistLookup.js, which
// lyricsLookup.js imports. The rest of that module looks artists up on
// Wikidata/MusicBrainz, which the app does not do.
const LATIN_ACCENTS = new RegExp(`[${String.fromCodePoint(0x300)}-${String.fromCodePoint(0x36f)}]`, "gu");

/** A name reduced to what tells it apart: no case, accents, punctuation or spaces. */
export function artistNameKey(name) {
  return String(name || "")
    .normalize("NFKD")
    .replace(LATIN_ACCENTS, "")
    .toLowerCase()
    .replace(/[\p{P}\p{S}\s]+/gu, "");
}
