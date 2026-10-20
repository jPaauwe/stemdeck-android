// Modified for StemDeck for Android (unofficial on-device port): the backend
// this page talks to runs inside the app on the phone. Changes from upstream
// are marked "Android:". Original: static/mobile/app.js in
// github.com/stemdeckapp/stemdeck (Apache-2.0), see ../UPSTREAM_COMMIT.
//
// StemDeck mobile — vanilla port of the Claude Design prototype
// (design/mobile/StemDeck-Mobile.dc.html). No framework, no build step,
// to match the rest of static/. Library + Mixer are wired to the real API
// (GET /api/jobs, /api/jobs/{id}, the Web Audio engine, mixdown export).
// Extract is still mock pending the SSE/upload wiring (next step).
import { fetchJobs, jobToCard } from "../js/shared/jobs.js";
import { createChunkedAudioEngine } from "../js/chunkedAudioEngine.js";
import { PITCH_MAX, PITCH_MIN, clampPitch } from "../js/pitchBus.js";
import { createPlaybackContext } from "../js/audioContext.js";
// Android: the click track, desktop's own scheduler (static/js/metronome.js, unchanged).
import { createMetronome, computeCountIn } from "../js/metronome.js";
// Per-stem label + color, keyed by the backend stem name. Unknown names fall
// back to a rotating palette so non-standard models still render sensibly.
const STEM_META = {
  vocals: { label: "Vocals", color: "#f0506e" },
  drums: { label: "Drums", color: "#f5862b" },
  bass: { label: "Bass", color: "#f5c518" },
  guitar: { label: "Guitar", color: "#3fcf6e" },
  piano: { label: "Piano", color: "#9b6cf0" },
  other: { label: "Other", color: "#4a9bf5" },
  // On-demand lead/backing vocal split (#275) -- not one of the Extract chips
  // (see vocalSplitMode below), just lane metadata for once a job has one.
  lead_vocals: { label: "Lead Vocals", color: "#f0506e" },
  backing_vocals: { label: "Backing Vocals", color: "#c44ad0" },
};
const FALLBACK_COLORS = ["#f0506e", "#f5862b", "#f5c518", "#3fcf6e", "#9b6cf0", "#4a9bf5", "#2bd4c4", "#c44ad0"];
function stemMeta(name, idx) {
  return STEM_META[name] || { label: name.charAt(0).toUpperCase() + name.slice(1), color: FALLBACK_COLORS[idx % FALLBACK_COLORS.length] };
}

// Stem chips shown on the Extract screen. lead_vocals/backing_vocals aren't
// separately selectable here -- they're an on-demand refinement of the
// Vocals chip (see the vocalSplitMode segmented control in extractScreen()).
const EXTRACT_BASE_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"];
const EXTRACT_STEMS = EXTRACT_BASE_STEMS.map((id) => ({ id, name: STEM_META[id].label, color: STEM_META[id].color }));

const DEFAULT_GRADIENT = "linear-gradient(150deg,#3a3a42,#202026)";
const FILTERS = ["All", "Favorites"];

// ─── Web Audio engine (reused from desktop: ../js/audioEngine.js) ───
let engine = null;
let engineTrackId = null;
let metronome = null; // Android: the click track for the current engine
let countInGeneration = 0; // Android: bumped by pause/stop, so a pending count-in is dropped
let engineToken = 0; // guards against races when switching tracks mid-decode
let engineReady = false; // true once the current engine has decodable stems
let pendingPlay = false;

// Shared AudioContext, created + resumed inside a user gesture (mobile/iOS
// won't start audio otherwise). Once running it stays running, so later
// programmatic play() calls work even outside a gesture.
let audioCtx = null;
function ensureAudioCtx() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  if (!audioCtx) audioCtx = createPlaybackContext(AC);
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  return audioCtx;
}


const ICON = {
  chevron: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>',
  dots: '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>',
  prev: '<svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor"><path d="M7 6v12M19 6l-9 6 9 6z"/></svg>',
  next: '<svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor"><path d="M17 6v12M5 6l9 6-9 6z"/></svg>',
  play: (sz, fill) => `<svg width="${sz}" height="${sz}" viewBox="0 0 24 24" fill="${fill}"><path d="M8 5.5v13l11-6.5z"/></svg>`,
  pause: (sz, fill) => `<svg width="${sz}" height="${sz}" viewBox="0 0 24 24" fill="${fill}"><rect x="6" y="5" width="4" height="14" rx="1.3"/><rect x="14" y="5" width="4" height="14" rx="1.3"/></svg>`,
  download: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M8 11l4 4 4-4"/><path d="M4 21h16"/></svg>',
  search: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#65656d" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg>',
  link: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#65656d" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/></svg>',
  upload: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="M8 8l4-4 4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
  heart: (on) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="${on ? "currentColor" : "none"}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>`,
  check: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>',
  scissors: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4L8.12 15.88M14.47 14.48L20 20M8.12 8.12L12 12"/></svg>',
  tabLib: '<svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
  tabMix: '<svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 8h16M4 16h16"/><circle cx="9" cy="8" r="2.4" fill="currentColor" stroke="none"/><circle cx="15" cy="16" r="2.4" fill="currentColor" stroke="none"/></svg>',
  tabExt: '<svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.6"/><circle cx="6" cy="18" r="2.6"/><path d="M20 4L8.5 15.5M14 14.5L20 20M8.5 8.5L11.5 11.5"/></svg>',
};

// Android: the app opens the Library (#library) after a song is shared to it.
const INITIAL_TAB = ["library", "extract", "mixer"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "mixer";

// Android: stem file formats. The choice applies to new songs and to "Export mix".
const STEM_FORMATS = [
  { id: "mp3", name: "MP3", note: "320 kbps · 2.4 MB a minute" },
  { id: "wav", name: "WAV", note: "Lossless · 10 MB a minute" },
];
// Android: where the separation runs. The GPU is used when it works;
// the engine falls back to the CPU on its own.
const PROCESSORS = [
  { id: "gpu", name: "GPU + CPU", note: "Every core plus the GPU" },
  { id: "cpu", name: "CPU only", note: "Every core" },
];
fetch("/api/settings").then((r) => r.json()).then((s) => {
  if (s && (s.stem_format === "mp3" || s.stem_format === "wav")) state.stemFormat = s.stem_format;
  if (s && typeof s.gpu === "boolean") state.processor = s.gpu ? "gpu" : "cpu";
  // Android: the click track's settings, kept by the app (see Settings.kt).
  const c = s && s.click;
  if (c && typeof c === "object") {
    if (typeof c.on === "boolean") state.click.on = c.on;
    if (typeof c.volume === "number") state.click.volume = Math.max(0, Math.min(1, c.volume));
    if (Number.isInteger(c.accent)) state.click.accent = c.accent;
    if (Number.isInteger(c.count_in)) state.click.countIn = c.count_in;
    applyClickSettings();
  }
  // Android: as desktop's wireAutoSectionsToggle, a sections choice left on by
  // an earlier session does not survive into this one.
  if (s && s.auto_sections) saveAutoSections(false);
  render();
}).catch(() => {});
// Android: which models the first song will download (the Extract screen says so).
fetch("/api/config").then((r) => r.json()).then((c) => {
  if (c && c.models) { state.models = c.models; if (state.tab === "extract") render(); }
}).catch(() => {});

const state = {
  tab: INITIAL_TAB,
  mixerView: "stems",
  playing: false,
  progress: 0,
  vols: {}, // per-stem gain 0..1, keyed by backend stem name
  muted: {},
  solo: {},
  speed: 1.0,
  pitch: 0, // global transpose in semitones, -6..+6
  selected: { vocals: true, drums: true, bass: true, guitar: true, piano: true, other: true },
  stemFormat: "mp3", // Android: "mp3" (320 kbps) or "wav"; loaded from /api/settings
  processor: "gpu", // Android: "gpu" or "cpu"; loaded from /api/settings
  quality: "High",
  filter: "All",
  // Library data from /api/jobs. libState: "loading" | "ready" | "empty" | "error".
  tracks: [],
  libState: "loading",
  swipedTrackId: null, // track whose swipe-to-delete is revealed
  current: null, // selected track card (with .detail once loaded), or null
  // Extract screen.
  extractUrl: "",
  extractFile: null, // File chosen for upload
  extractJob: null, // { id, title, status, progress, stage, vocalSplitMode } while a job runs
  // "all" = plain Vocals lane (default); "split" = also run the on-demand
  // lead/backing split (#275) once the base separation finishes.
  vocalSplitMode: "all",
  // Android: the click track. `mult` (half/double time) is per song, as on desktop.
  click: { on: false, volume: 0.6, accent: -1, countIn: 0, mult: 1 },
  exportClick: false, // include the click in "Export mix" (opt-in, as on desktop)
  models: null, // { separation, beat, sections }: downloaded yet?
  // Android: find song sections for the next import (desktop's auto_sections;
  // off at every start and after each import, as on desktop).
  autoSections: false,
  loopSection: null, // Android: id of the section the transport loops, or null
};

let extractES = null; // EventSource for the active extraction
let extractPoll = null; // REST poll fallback timer

const app = document.getElementById("app");

function fmt(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2, "0");
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// Cover/thumbnail art. Uses the real YouTube/SoundCloud thumbnail when present
// (layered over the gradient as a fallback if it fails to load); otherwise the
// gradient + initial. The URL is constrained to a clean https URL so it can't
// break out of the CSS url().
function safeThumb(url) {
  return typeof url === "string" && /^https:\/\/[^"'()\\\s]+$/.test(url) ? url : "";
}
function artStyle(card) {
  const g = (card && card.gradient) || DEFAULT_GRADIENT;
  const t = card && safeThumb(card.thumb);
  return t
    ? `background-image:url('${t}'), ${g};background-size:cover;background-position:center;`
    : `background:${g};`;
}
function artLabel(card) {
  return card && safeThumb(card.thumb) ? "" : esc((card && card.initial) || "♪");
}

// ─── Mixer / engine helpers ───
function lanes() {
  return state.current?.detail?.lanes || [];
}
function curDuration() {
  return state.current?.detail?.duration || 0;
}
function anySolo() {
  return Object.values(state.solo).some(Boolean);
}
function laneActive(name) {
  const muted = !!state.muted[name];
  const soloed = !!state.solo[name];
  return !(muted || (anySolo() && !soloed));
}
function effectiveGain(name) {
  return laneActive(name) ? (state.vols[name] ?? 1) : 0;
}
function applyMix() {
  if (!engine) return;
  for (const l of lanes()) engine.setGain(l.name, effectiveGain(l.name));
}

// ─── Android: click track ───
// The beat grid comes from stems/beats.json (made after separation, as on
// desktop); the scheduler is desktop's metronome.js, which places each click
// in the engine's source time, so it follows speed changes and seeks.

// Where beat `i` sits in its bar, from the detected bar marks
// (desktop's beatgrid.js barPositionIndex).
function barPosition(i, bars) {
  if (!bars || !bars.length) return null;
  let mark = null;
  for (const b of bars) {
    if (b.beat <= i) mark = b;
    else break;
  }
  if (!mark || !Number.isInteger(mark.beats_per_bar) || mark.beats_per_bar < 1) return null;
  return [(i - mark.beat) % mark.beats_per_bar, mark.beats_per_bar];
}

function currentGrid() {
  const g = state.current?.grid;
  return g && Array.isArray(g.beats) && g.beats.length ? g : null;
}

// Desktop's applyMetronomeAccent: Auto follows the detected bar marks;
// an explicit meter (or Off) counts from the first beat.
function applyClickAccent() {
  if (!metronome) return;
  const bars = currentGrid()?.bars || [];
  metronome.setGrouping?.(null);
  if (state.click.accent < 0 && bars.length) {
    metronome.setDownbeatFn?.((i) => barPosition(i, bars)?.[0] === 0);
    metronome.setBarPositionFn?.((i) => barPosition(i, bars));
  } else {
    metronome.setDownbeatFn?.(null);
    metronome.setBarPositionFn?.(null);
    metronome.setBeatsPerBar?.(Math.max(0, state.click.accent));
  }
}

function applyClickSettings() {
  if (!metronome) return;
  metronome.setVolume(state.click.volume);
  metronome.setMultiplier(state.click.mult);
  applyClickAccent();
  metronome.setEnabled(state.click.on);
}

// Called when the engine is ready and when the grid arrives, in either order.
function setupClick() {
  const grid = currentGrid();
  if (metronome || !grid || !engine || !engineReady || engineTrackId !== state.current?.id) return;
  metronome = createMetronome(engine, grid.beats, { volume: state.click.volume });
  applyClickSettings();
}

function teardownClick() {
  countInGeneration++;
  if (metronome) { metronome.destroy(); metronome = null; }
}

function saveClickSettings() {
  const { on, volume, accent, countIn } = state.click;
  fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ click: { on, volume, accent, count_in: countIn } }),
  }).catch(() => {});
}

// Desktop's _armCountIn: start the audio late (engine.play(leadIn)) and play
// the count clicks in the gap, whether or not the running click is on.
// Returns true when it started playback itself.
function armCountIn(startPos) {
  const grid = currentGrid();
  if (state.click.countIn < 1 || !engine?.supportsCountIn || !metronome || !grid) return false;
  const { leadIn, clicks } = computeCountIn(grid.beats, grid.bars || [], {
    countBars: state.click.countIn,
    multiplier: metronome.getMultiplier?.() ?? 1,
    accentMode: state.click.accent,
    start: startPos,
  });
  if (leadIn <= 0 || !clicks.length) return false;
  const sourceClicks = clicks.map((c) => ({ time: startPos - leadIn + c.offset, level: c.level, accent: c.accent }));
  const generation = ++countInGeneration;
  const clickTrack = metronome;
  engine.play(leadIn).then((started) => {
    if (!started || generation !== countInGeneration) return;
    clickTrack.playCountIn(sourceClicks);
  });
  return true;
}

// Android: the sections choice for the next import lives on the server, which
// is what the job reads when it is created (desktop's auto_sections).
function saveAutoSections(next) {
  state.autoSections = next;
  return fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ auto_sections: next }) })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
    .then((d) => { state.autoSections = !!d.auto_sections; })
    .catch(() => { state.autoSections = !next; if (next) toast("Couldn't save that."); })
    .finally(() => { if (state.tab === "extract") render(); });
}

// Android: desktop's section names (static/js/sections.js sectionDisplayName):
// the kind's name, numbered when the song has more than one of that kind.
const SECTION_NAMES = {
  intro: "Intro", outro: "Outro", break: "Break", bridge: "Bridge", inst: "Instrumental",
  solo: "Solo", verse: "Verse", chorus: "Chorus", part: "Part",
};
function sectionName(sec, all) {
  const kind = String(sec?.kind || "").toLowerCase();
  const name = SECTION_NAMES[kind];
  if (!name) return String(sec?.name || "");
  const peers = [...all].sort((a, b) => a.start - b.start).filter((x) => String(x?.kind || "").toLowerCase() === kind);
  if (peers.length < 2) return name;
  const pos = peers.findIndex((x) => x.id === sec.id);
  return pos < 0 ? name : `${name} ${pos + 1}`;
}

function currentSections() {
  const list = state.current?.detail?.sections;
  return Array.isArray(list) && list.length ? list : null;
}

// Android: the song's sections as coloured blocks over the waveform. Tap one to
// play from its start; hold one to loop it (hold it again to stop looping).
function sectionStrip() {
  const list = currentSections();
  const dur = curDuration();
  if (!list || !dur) return "";
  return `<div class="sec-strip" data-sections>${[...list].sort((a, b) => a.start - b.start).map((x) => {
    const left = (Math.max(0, x.start) / dur) * 100;
    const width = (Math.max(0, Math.min(dur, x.end) - Math.max(0, x.start)) / dur) * 100;
    const looped = state.loopSection === x.id;
    return `<div class="sec-block${looped ? " looped" : ""}" data-sec="${esc(x.id)}" style="left:${left.toFixed(4)}%;width:${width.toFixed(4)}%;--sc:${esc(x.color || "#8391a5")}" role="button" aria-label="${esc(sectionName(x, list))}${looped ? ", looping" : ""}"><span>${looped ? "⟲ " : ""}${esc(sectionName(x, list))}</span></div>`;
  }).join("")}</div>`;
}

function setSectionLoop(sec) {
  if (!engine) return;
  if (!sec || state.loopSection === sec.id) {
    state.loopSection = null;
    engine.setLoop(false, 0, 0);
    toast("Loop off");
  } else {
    state.loopSection = sec.id;
    engine.setLoop(true, sec.start, sec.end);
    engine.seek(sec.start);
    toast(`Looping ${sectionName(sec, currentSections() || [])}`);
  }
  render();
}

// Beats and bar starts as a strip under the waveform (detected grid).
function beatStrip() {
  const g = currentGrid();
  const dur = curDuration();
  if (!g || !dur) return "";
  const x = (t) => ((t / dur) * 1000).toFixed(1);
  let beats = "";
  for (const t of g.beats) beats += `M${x(t)} 5V12`;
  let bars = "";
  const marks = g.bars || [];
  for (let m = 0; m < marks.length; m++) {
    const end = m + 1 < marks.length ? marks[m + 1].beat : g.beats.length;
    const n = marks[m].beats_per_bar;
    if (!(n >= 1)) continue;
    for (let i = marks[m].beat; i < end; i += n) if (i >= 0 && i < g.beats.length) bars += `M${x(g.beats[i])} 0V12`;
  }
  return `<svg class="beat-strip" viewBox="0 0 1000 12" preserveAspectRatio="none" aria-hidden="true">` +
    `<path d="${beats}" class="bs-beat"/><path d="${bars}" class="bs-bar"/></svg>`;
}

const ACCENTS = [
  [-1, "Accent: auto"], [0, "Accent: off"], [2, "2/4"], [3, "3/4"], [4, "4/4"], [5, "5/4"],
  [6, "6/8"], [7, "7/8"], [9, "9/8"], [12, "12/8"],
];
const COUNT_INS = [[0, "Count-in: off"], [1, "Count-in: 1 bar"], [2, "Count-in: 2 bars"], [3, "Count-in: 3 bars"], [4, "Count-in: 4 bars"]];

function clickRows() {
  const g = currentGrid();
  const c = state.click;
  if (!g) {
    const why = state.current?.detail?.notes?.beats;
    return why ? `<div class="click-note">${esc(why)}</div>` : "";
  }
  const pct = Math.round(c.volume * 100);
  const opt = (list, v) => list.map(([val, label]) => `<option value="${val}"${val === v ? " selected" : ""}>${label}</option>`).join("");
  // An explicit meter the presets do not have (set on another device) still shows.
  const accents = ACCENTS.some(([v]) => v === c.accent) ? ACCENTS : [...ACCENTS, [c.accent, `${c.accent} beats`]];
  const mult = metronome?.getMultiplier?.() ?? c.mult;
  return `<div class="speed-row click-row">
        <span class="speed-row-label">Click</span>
        <button class="click-toggle${c.on ? " on" : ""}" data-action="click-toggle" aria-pressed="${c.on}">${c.on ? "On" : "Off"}</button>
        <input type="range" class="speed-slider" data-click-vol min="0" max="1" step="0.05" value="${c.volume}" aria-label="Click volume">
        <span class="speed-row-val">${pct}%</span>
      </div>
      <div class="click-opts">
        <select class="click-select" data-click-accent aria-label="Accent">${opt(accents, c.accent)}</select>
        <div class="segmented sm click-rate" role="radiogroup" aria-label="Click rate">${[[0.5, "½×"], [1, "1×"], [2, "2×"]]
          .map(([m, l]) => `<button class="${mult === m ? "on" : ""}" data-action="click-mult" data-mult="${m}" role="radio" aria-checked="${mult === m}">${l}</button>`).join("")}</div>
        <select class="click-select" data-click-countin aria-label="Count-in">${opt(COUNT_INS, c.countIn)}</select>
      </div>`;
}

function cap(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}
function drLabel(dr) {
  if (dr < 7) return "Compressed";
  if (dr < 10) return "Moderate";
  if (dr < 14) return "High";
  return "Wide";
}
function stabilityLabel(pct) {
  if (pct >= 90) return "Very stable";
  if (pct >= 70) return "Stable";
  if (pct >= 40) return "Variable";
  return "Unstable";
}

// Build the Analysis tab's stat cards + stem-presence rows from a job detail.
function extractAnalysis(d, laneList) {
  const stats = [];
  if (d.key) stats.push({ k: "KEY", v: d.key, s: d.scale ? cap(d.scale) : "" });
  if (d.bpm) stats.push({ k: "TEMPO", v: String(Math.round(d.bpm)), s: "BPM" });
  if (d.lufs != null) stats.push({ k: "LUFS", v: Number(d.lufs).toFixed(1), s: d.peak_db != null ? `Peak ${Number(d.peak_db).toFixed(1)} dB` : "Integrated" });
  if (d.duration) stats.push({ k: "DURATION", v: fmt(d.duration), s: "Full length" });
  if (d.dynamic_range != null) stats.push({ k: "DYN RANGE", v: Number(d.dynamic_range).toFixed(1), s: drLabel(d.dynamic_range) });
  if (d.tempo_stability != null) {
    const pct = d.tempo_stability > 1 ? d.tempo_stability : d.tempo_stability * 100;
    stats.push({ k: "STABILITY", v: Math.round(pct) + "%", s: stabilityLabel(pct) });
  }
  const sp = d.stem_presence || {};
  const presence = laneList
    .filter((l) => sp[l.name] != null)
    .map((l) => ({ name: l.label, color: l.color, val: Math.round(sp[l.name]) }));
  return { stats, presence };
}

// Repaint the main waveform's played (yellow) vs. remaining bars as playback
// advances. Only touches the DOM when the played-bar count actually changes
// (≤ N times over the whole track), not every animation frame.
let _lastPlayedBar = -1;
function paintWaveProgress() {
  const bars = app.querySelectorAll(".wave-bars > i");
  if (!bars.length) return;
  const played = Math.round(state.progress * bars.length);
  if (played === _lastPlayedBar) return;
  for (let i = 0; i < bars.length; i++) {
    bars[i].style.background = i <= played ? "#f5b417" : "#34343c";
  }
  _lastPlayedBar = played;
}

function onEngineTime(t) {
  const dur = curDuration() || 1;
  state.progress = Math.max(0, Math.min(1, t / dur));
  const head = app.querySelector(".playhead");
  const cur = app.querySelector(".wave-times .cur");
  if (head) head.style.left = state.progress * 100 + "%";
  if (cur) cur.textContent = fmt(t);
  paintWaveProgress();
}

// Load a track: fetch detail, build lanes, spin up the Web Audio engine.
async function openTrack(card, { autoplay = false } = {}) {
  state.tab = "mixer";
  state.current = { ...card, detail: null, loading: true, error: null };
  state.playing = false;
  state.progress = 0;
  state.speed = 1.0;
  state.pitch = 0;
  render();

  teardownClick(); // Android
  if (engine) { engine.destroy(); engine = null; engineTrackId = null; }
  engineReady = false;
  const token = ++engineToken;
  pendingPlay = autoplay;
  state.click.mult = 1; // Android: half/double time is per song, as on desktop
  state.loopSection = null; // Android: a new song starts without a loop

  // Android: the click track's beat grid, alongside the detail (404 when the
  // song has none, e.g. separated before 1.1).
  state.current.grid = null;
  fetch(`/api/jobs/${card.id}/stems/beats.json`, { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then((g) => {
      if (token !== engineToken || !g || !state.current) return;
      state.current.grid = g;
      setupClick();
      render();
    })
    .catch(() => {});

  let detail;
  try {
    const res = await fetch(`/api/jobs/${card.id}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`GET /api/jobs/${card.id} -> ${res.status}`);
    detail = await res.json();
  } catch (e) {
    if (token !== engineToken) return;
    console.warn("[mobile] track detail failed:", e);
    state.current.loading = false;
    state.current.error = "Couldn't load this track.";
    render();
    return;
  }
  if (token !== engineToken) return;

  // Use WAV stems with range-request chunking (chunkedAudioEngine.js): fetches
  // 10-second windows at a time, so the first audio starts after ~7 MB instead
  // of the full file, and peak RAM stays around 28 MB regardless of track length.
  // Once the on-demand lead/backing split (#275) has run, show those two
  // lanes in place of the plain Vocals lane rather than all three -- they're
  // a decomposition of the same signal (mirrors the backend's mixdown guard
  // in app/api/stems.py), not three independent tracks.
  const stemNames = new Set((detail.stems || []).map((s) => s && s.name));
  const vocalSplitDone = stemNames.has("lead_vocals") && stemNames.has("backing_vocals");
  const laneList = (detail.stems || [])
    .filter((s) => s && s.url && s.name !== "original" && !(vocalSplitDone && s.name === "vocals"))
    .map((s, i) => ({ name: s.name, url: s.url, ...stemMeta(s.name, i) }));

  state.vols = {};
  state.muted = {};
  state.solo = {};
  for (const l of laneList) state.vols[l.name] = 1;

  state.current.detail = {
    duration: detail.duration || 0,
    lanes: laneList,
    hasVideo: !!detail.has_video,
    analysis: extractAnalysis(detail, laneList),
    notes: detail.analysis_notes || {}, // Android: why the beat grid or key is missing
    sections: Array.isArray(detail.sections) ? detail.sections : null, // Android: automatic sections
  };
  state.current.loading = false;
  state.current.stemCount = laneList.length;

  if (!laneList.length) {
    state.current.error = "This track has no playable stems yet.";
    render();
    return;
  }

  const engineOpts = {
    onTime: onEngineTime,
    onEnded: () => { state.playing = false; render(); },
    context: ensureAudioCtx(),
  };
  engine = createChunkedAudioEngine(laneList.map((l) => ({ name: l.name, url: l.url })), engineOpts);
  engineTrackId = card.id;
  render();

  const ok = await engine.ready;
  if (token !== engineToken) return;
  if (!ok) {
    state.current.error = "Couldn't load this track's audio.";
    pendingPlay = false;
    toast("Couldn't load this track's audio.");
    render();
    return;
  }
  engineReady = true;
  applyMix();
  setupClick(); // Android
  render();
  if (pendingPlay) { pendingPlay = false; engine.play(); state.playing = engine.isPlaying(); render(); }
}

function prevTrack() {
  if (!state.tracks.length || !state.current) return;
  const idx = state.tracks.findIndex((t) => t.id === state.current.id);
  const target = state.tracks[idx - 1];
  if (target) openTrack(target, { autoplay: state.playing });
}

function nextTrack() {
  if (!state.tracks.length || !state.current) return;
  const idx = state.tracks.findIndex((t) => t.id === state.current.id);
  const target = state.tracks[idx + 1];
  if (target) openTrack(target, { autoplay: state.playing });
}

function togglePlay() {
  if (!state.current) return;
  ensureAudioCtx(); // unlock audio within this gesture
  // Engine not loaded for this track yet → load then autoplay.
  if (!engine || engineTrackId !== state.current.id) {
    openTrack(state.current, { autoplay: true });
    return;
  }
  // Decode still in flight → remember the intent and play once ready.
  if (!engineReady) {
    pendingPlay = true;
    return;
  }
  if (engine.isPlaying()) {
    engine.pause();
    countInGeneration++; // Android: drop a count-in still waiting to start
    metronome?.cancelCountIn?.();
  } else {
    // Android: count in, leading into where playback starts (desktop's transport).
    let startPos = engine.getCurrentTime?.() ?? 0;
    const dur = curDuration();
    if (dur > 0 && startPos >= dur) startPos = 0;
    if (!armCountIn(startPos)) engine.play();
  }
  state.playing = engine.isPlaying();
  render();
}

function seekToFraction(frac) {
  if (!engine) return;
  engine.seek(Math.max(0, Math.min(1, frac)) * curDuration());
}

function triggerDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function exportMix(kind) {
  const id = state.current?.id;
  if (!id || !lanes().length) return;
  const names = [];
  const gains = [];
  for (const l of lanes()) {
    const g = effectiveGain(l.name);
    if (g > 0) { names.push(l.name); gains.push(g); }
  }
  if (!names.length) { toast("Every stem is muted — nothing to export."); return; }
  const q = new URLSearchParams({ stems: names.join(","), gains: gains.map((g) => g.toFixed(3)).join(",") });
  // Android: the click, as the user monitors it (desktop's _clickParams).
  if (state.exportClick && currentGrid()) {
    q.set("click", "1");
    q.set("click_mult", String(metronome?.getMultiplier?.() ?? state.click.mult));
    q.set("click_accent", String(state.click.accent));
    q.set("click_gain", state.click.volume.toFixed(3));
  }
  const path = kind === "mp4" ? `video.mp4` : `mixdown.wav`;
  triggerDownload(`/api/jobs/${id}/${path}?${q}`);
}

// Android: the stems the Mixer has active (not muted or soloed out), in one
// ZIP named after the song (desktop's stems/all.zip).
function exportStemsZip() {
  const id = state.current?.id;
  if (!id || !lanes().length) return;
  const names = lanes().filter((l) => effectiveGain(l.name) > 0).map((l) => l.name);
  if (!names.length) { toast("Every stem is muted — nothing to export."); return; }
  triggerDownload(`/api/jobs/${id}/stems/all.zip?${new URLSearchParams({ stems: names.join(",") })}`);
}

let _toastTimer = null;
function toast(msg) {
  let el = document.getElementById("m-toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "m-toast";
    el.className = "m-toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 2400);
}

function mainWaveform() {
  const N = 56;
  const played = Math.round(state.progress * N);
  let bars = "";
  for (let i = 0; i < N; i++) {
    let h = 24 + Math.abs(Math.sin(i * 0.7) + Math.sin(i * 0.21) * 0.7) * 46 + ((i * 37) % 11);
    h = Math.max(14, Math.min(98, h));
    const bg = i <= played ? "#f5b417" : "#34343c";
    bars += `<i style="height:${h}%;background:${bg}"></i>`;
  }
  return bars;
}

function stemWave(idx, color, vol, eff) {
  const wn = 28;
  const wplayed = Math.round(state.progress * wn);
  let out = "";
  for (let i = 0; i < wn; i++) {
    let wh = 16 + Math.abs(Math.sin(i * (0.55 + idx * 0.13)) + Math.sin(i * 0.33 + idx) * 0.6) * 62 + ((i * (7 + idx * 3)) % 9);
    wh = Math.max(7, Math.min(98, wh)) * (0.3 + 0.7 * vol);
    const wc = eff ? "#34343c" : i <= wplayed ? color : color + "3d";
    out += `<i style="height:${wh}%;background:${wc}"></i>`;
  }
  return out;
}

function stemsBody() {
  const c = state.current;
  if (!c) return `<div class="lib-note">Pick a track from your Library to start mixing.</div>`;
  if (c.loading) return `<div class="lib-note">Loading stems…</div>`;
  if (c.error) return `<div class="lib-note">${esc(c.error)}</div>`;
  const ll = lanes();
  if (!ll.length) return `<div class="lib-note">No stems available.</div>`;
  return `<div class="stems-grid">${ll.map((l, idx) => {
    const vol = state.vols[l.name] ?? 1;
    const muted = !!state.muted[l.name];
    const soloed = !!state.solo[l.name];
    const eff = !laneActive(l.name);
    const pct = Math.round(vol * 100) + "%";
    const mOn = muted ? `background:${l.color};color:#15100a;border-color:${l.color};` : "";
    const sOn = soloed ? "background:#f5b417;color:#1a1206;border-color:#f5b417;" : "";
    return `<div class="stem">
      <div class="stem-top">
        <div class="stem-dot" style="background:${l.color};box-shadow:0 0 8px ${l.color}88;opacity:${eff ? 0.4 : 1}"></div>
        <div class="stem-name" style="color:${eff ? "#6a6a72" : l.color}">${esc(l.label)}</div>
        <button class="ms-btn" style="${mOn}" data-action="mute" data-id="${l.name}">M</button>
        <button class="ms-btn" style="${sOn}" data-action="solo" data-id="${l.name}">S</button>
      </div>
      <div class="stem-wave">${stemWave(idx, l.color, vol, eff)}</div>
      <div class="fader" data-fader data-id="${l.name}">
        <div class="fader-track"></div>
        <div class="fader-fill" style="width:${pct};background:${eff ? "#46464e" : l.color}"></div>
        <div class="fader-knob" style="left:${pct};background:${eff ? "#8a8a90" : "#fff"}"></div>
      </div>
    </div>`;
  }).join("")}</div>`;
}

function analysisBody() {
  const a = state.current?.detail?.analysis;
  const hasVideo = state.current?.detail?.hasVideo;
  const statsHtml = a && a.stats.length
    ? `<div class="stats">${a.stats.map((x) => `<div class="stat"><div class="stat-k">${x.k}</div><div class="stat-v">${esc(x.v)}</div><div class="stat-s">${esc(x.s)}</div></div>`).join("")}</div>`
    : `<div class="lib-note" style="margin-top:8px">Analysis not available for this track.</div>`;
  const presenceHtml = a && a.presence.length
    ? `<div class="eyebrow">STEM PRESENCE</div>${a.presence.map((p) => `<div class="presence-row"><div class="presence-name">${esc(p.name)}</div><div class="presence-bar"><i style="width:${p.val}%;background:${p.color}"></i></div><span class="presence-val">${p.val}%</span></div>`).join("")}`
    : "";
  // Android: the click track's beat grid, or why the song has none.
  const g = currentGrid();
  const notes = state.current?.detail?.notes || {};
  let gridHtml = "";
  if (g) {
    const marks = g.bars || [];
    const meter = marks.length === 1 ? `${marks[0].beats_per_bar} beats a bar` : marks.length > 1 ? `meter changes ${marks.length - 1}×` : "no bar marks";
    gridHtml = `<div class="eyebrow">BEAT GRID</div><div class="grid-info">
      <div class="gi-main">${Number(g.bpm).toFixed(1)} BPM · ${g.confidence}% of beats on a drum hit</div>
      <div class="gi-sub">${g.beats.length} beats · ${esc(meter)} · from the ${esc(g.source)} by Beat This!</div></div>`;
  } else if (notes.beats) {
    gridHtml = `<div class="eyebrow">BEAT GRID</div><div class="grid-info"><div class="gi-sub">${esc(notes.beats)}</div></div>`;
  }
  const keyNote = (notes.key ? `<div class="grid-info" style="margin-top:8px"><div class="gi-sub">${esc(notes.key)}</div></div>` : "") +
    (notes.sections ? `<div class="grid-info" style="margin-top:8px"><div class="gi-sub">${esc(notes.sections)}</div></div>` : "");
  const clickOpt = g
    ? `<button class="check-row" data-action="export-click" aria-pressed="${state.exportClick}"><span class="check${state.exportClick ? " on" : ""}">${state.exportClick ? ICON.check : ""}</span>Include the click track</button>`
    : "";
  const exportBtns = `${clickOpt}<button class="cta sm" data-action="export" data-kind="wav">${ICON.download}Export mix</button>${
    hasVideo ? `<button class="cta sm" data-action="export" data-kind="mp4" style="margin-top:10px">${ICON.download}Export MP4 (with video)</button>` : ""
  }<button class="cta sm ghost" data-action="export-zip">${ICON.download}Export stems (ZIP)</button>`;
  return `<div class="pad" style="padding-top:16px">${statsHtml}${gridHtml}${keyNote}${presenceHtml}${exportBtns}</div>`;
}

/**
 * The global transpose row.
 *
 * Phones get this UI rather than the desktop one, which is where transpose
 * lives everywhere else, so without a control here the feature simply does not
 * exist on a phone however well the engine supports it. There are no per-lane
 * keys on this screen, so this is the whole of transpose on mobile: every lane
 * moves together.
 *
 * Steppers rather than a slider. A semitone is one twelfth of the range and a
 * slider that wide is not something a thumb can land on.
 */
function keyRow() {
  const n = state.pitch;
  const label = n === 0 ? "0" : (n > 0 ? `+${n}` : `${n}`);
  const off = !engineReady || engine?.supportsPitchShift?.() !== true;
  const why = off ? ' title="Transpose is not available on this connection"' : "";
  return `<div class="speed-row key-row"${why}>
        <span class="speed-row-label">Key</span>
        <div class="key-steps">
          <button class="key-step" data-key-step="-1" ${off || n <= PITCH_MIN ? "disabled" : ""}>&minus;</button>
          <button class="key-step" data-key-step="1" ${off || n >= PITCH_MAX ? "disabled" : ""}>+</button>
        </div>
        <span class="speed-row-val${off ? " off" : ""}">${label}</span>
      </div>`;
}

function mixerScreen() {
  const c = state.current || { title: "No track selected", sub: "Pick one from your Library", initial: "♪", gradient: DEFAULT_GRADIENT, stemCount: 0 };
  const sourceTag = c.sub || "—";
  const stemTag = c.stemCount ? `${c.stemCount} stems` : "";
  const dur = curDuration();
  const body = state.mixerView === "stems" ? stemsBody() : analysisBody();
  const ready = engineReady && engineTrackId === state.current?.id;
  const preparing = !!state.current && !state.current.loading && !state.current.error && !ready;
  const canPlay = ready;
  const curIdx = state.current ? state.tracks.findIndex((t) => t.id === state.current.id) : -1;
  const hasPrev = curIdx > 0;
  const hasNext = curIdx >= 0 && curIdx < state.tracks.length - 1;

  return `<div class="screen scrl">
    <div class="pad">
      <div class="mx-head">
        <button class="icon-btn" data-action="tab" data-tab="library">${ICON.chevron}</button>
        <span class="now-playing">NOW PLAYING</span>
        <button class="icon-btn">${ICON.dots}</button>
      </div>
      <div class="cover-wrap">
        <div class="cover" style="${artStyle(c)}"><span>${artLabel(c)}</span></div>
        <div class="track-title">${esc(c.title)}</div>
        <div class="track-sub">${esc(c.sub)}</div>
        <div class="tags"><span class="tag">${esc(sourceTag)}</span>${stemTag ? `<span class="tag">${stemTag}</span>` : ""}</div>
      </div>
      <div class="wave">
        ${sectionStrip() /* Android */}
        <div class="wave-bars" data-seek>${mainWaveform()}<div class="playhead" style="left:${state.progress * 100}%"></div></div>
        ${beatStrip() /* Android */}
        <div class="wave-times"><span class="cur">${fmt(state.progress * dur)}</span><span class="dur">${fmt(dur)}</span></div>
      </div>
      <div class="transport">
        <button class="t-step" data-action="prev" ${hasPrev ? "" : "disabled"}>${ICON.prev}</button>
        <button class="t-play" data-action="play" data-playing="${state.playing}" ${canPlay ? "" : "disabled style=opacity:.45"}>${state.playing ? ICON.pause(26, "#1a1206") : ICON.play(28, "#1a1206")}</button>
        <button class="t-step" data-action="next" ${hasNext ? "" : "disabled"}>${ICON.next}</button>
      </div>
      <div class="speed-row">
        <span class="speed-row-label">Speed</span>
        <input type="range" class="speed-slider" data-speed min="0" max="2" step="0.25" value="${state.speed}">
        <span class="speed-row-val">${state.speed % 1 === 0 ? state.speed.toFixed(1) : state.speed}x</span>
      </div>
      ${keyRow()}
      ${clickRows() /* Android */}
      ${preparing ? '<div class="mx-prep">Preparing audio…</div>' : ""}
      <div class="segmented">
        <button class="${state.mixerView === "stems" ? "on" : ""}" data-action="mixview" data-view="stems">Stems</button>
        <button class="${state.mixerView === "analysis" ? "on" : ""}" data-action="mixview" data-view="analysis">Analysis</button>
      </div>
    </div>
    ${body}
  </div>`;
}

function libraryBody() {
  if (state.libState === "loading") {
    return `<div class="lib-note">Loading your library…</div>`;
  }
  if (state.libState === "error") {
    return `<div class="lib-note">Couldn't reach the server. <button class="lib-retry" data-action="reload">Retry</button></div>`;
  }
  if (state.libState === "empty") {
    return `<div class="lib-note">No tracks yet. Head to <b>Extract</b> to split your first song.</div>`;
  }
  // The Favorites chip used to set a filter nothing read, so it listed every
  // track (#734). Favourites are on the server now, so it can filter for real.
  const favOnly = state.filter === "Favorites";
  const shown = favOnly ? state.tracks.filter((t) => t.favorite) : state.tracks;
  if (favOnly && !shown.length) {
    return `<div class="lib-note">No favourites yet. Tap the heart on a track to add it here.</div>`;
  }
  return `<div class="eyebrow">${favOnly ? "FAVORITES" : "RECENT"}</div>
    ${shown.map((t) => {
      const unavailable = t.status === "unavailable";
      const canReimport = unavailable && t.sourceUrl && !t.sourceUrl.startsWith("local:");
      const infoHtml = unavailable
        ? `<div class="t">${esc(t.title)}</div><div class="s track-warn">${
          canReimport ? "Track unavailable · tap to reimport" : "Track unavailable · re-upload to restore"
        }</div>`
        : `<div class="t">${esc(t.title)}</div><div class="s">${esc(t.sub)}</div><div class="m">${esc(t.meta)}</div>`;
      return `<div class="track-wrap${state.swipedTrackId === t.id ? " swiped" : ""}">
      <button class="track-delete" data-action="delete" data-id="${esc(t.id)}">Delete</button>
      <div class="track" data-action="${unavailable ? "reimport" : "open"}" data-id="${esc(t.id)}">
        <div class="track-art" style="${artStyle(t)}">${artLabel(t)}</div>
        <div class="track-info">${infoHtml}</div>
        <div class="track-dot ${t.status}"></div>
        <button class="track-fav${t.favorite ? " on" : ""}" data-action="favorite" data-id="${esc(t.id)}" aria-pressed="${t.favorite}" aria-label="${t.favorite ? "Remove from Favorites" : "Add to Favorites"}">${ICON.heart(t.favorite)}</button>
        <button class="track-load" data-action="${unavailable ? "reimport" : "open"}" data-id="${esc(t.id)}">${unavailable ? "Fix" : "Load"}</button>
      </div>
    </div>`;
    }).join("")}`;
}

function libraryScreen() {
  return `<div class="screen scrl">
    <div class="pad">
      <div class="lib-head">
        <span class="h1">Library</span>
        <div class="avatar">JS</div>
      </div>
      <div class="search">${ICON.search}<span>Search your library</span></div>
      <div class="filters scrl">${FILTERS.map((f) => `<span class="filter ${f === state.filter ? "on" : ""}" data-action="filter" data-filter="${f}">${f}</span>`).join("")}</div>
      ${libraryBody()}
    </div>
  </div>`;
}

function extractProgressCard() {
  const j = state.extractJob;
  if (!j) return "";
  const pct = Math.round((j.progress || 0) * 100);
  const failed = j.status === "error";
  const cancelled = j.status === "cancelled";
  // Trashed while it ran (#748): halted, its files kept for the desktop's Trash.
  const stopped = j.status === "stopped";
  const done = j.status === "done";
  const line = failed ? "Failed" : cancelled ? "Cancelled" : stopped ? "Stopped" : done ? "Done" : `${j.stage || j.status}…`;
  return `<div class="eyebrow">IN PROGRESS</div>
    <div class="progress-card">
      <div class="progress-top">
        <div class="progress-art"></div>
        <div style="flex:1;min-width:0">
          <div style="font-size:14px;font-weight:600;color:#eaeaec;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(j.title || "Track")}</div>
          <div style="font-size:12px;color:${failed ? "#f0506e" : "#7a7a82"};margin-top:1px">${esc(line)}</div>
        </div>
        <span class="progress-pct">${done ? "100" : pct}%</span>
      </div>
      <div class="progress-bar"><i style="width:${done ? 100 : pct}%"></i></div>
    </div>`;
}

function extractScreen() {
  const fileName = state.extractFile?.name;
  return `<div class="screen scrl">
    <div class="pad">
      <div class="h1">Extract stems</div>
      <div class="sub">Upload audio to split it into stems on this phone.</div>
      <button class="upload" data-action="pick-file">${ICON.upload}${fileName ? esc(fileName) : "Upload audio file"}</button>
      <input id="ext-file" type="file" accept="audio/*,video/mp4,.mp3,.wav,.flac,.m4a,.ogg,.opus" style="display:none">
      <div class="eyebrow">STEMS TO EXTRACT</div>
      <div class="chips">${EXTRACT_STEMS.map((s) => {
        const on = !!state.selected[s.id];
        const onStyle = on ? `border-color:${s.color};background:${s.color}1c;` : "";
        return `<button class="chip-btn ${on ? "on" : ""}" style="${onStyle}" data-action="chip" data-id="${s.id}"><div class="dot" style="background:${s.color}"></div><span class="nm">${s.name}</span>${on ? ICON.check : ""}</button>`;
      }).join("")}</div>
      <div class="vocal-mode-row">
        <span class="vocal-mode-label">Vocals</span>
        <div class="segmented sm">
          <button class="${state.selected.vocals && state.vocalSplitMode !== "split" ? "on" : ""}" data-action="vocalmode" data-mode="all">Combined</button>
          <button class="${state.selected.vocals && state.vocalSplitMode === "split" ? "on" : ""}" data-action="vocalmode" data-mode="split">Lead + Backing</button>
        </div>
      </div>
      <!-- Android: the file format for stems and mix exports, remembered by the app. -->
      <div class="eyebrow">FILE FORMAT</div>
      <div class="chips">${STEM_FORMATS.map((f) => {
        const on = state.stemFormat === f.id;
        return `<button class="chip-btn ${on ? "on" : ""}" style="${on ? "border-color:rgba(255,255,255,.45);background:rgba(255,255,255,.06);" : ""}" data-action="format" data-id="${f.id}"><span class="nm">${f.name}<br><span style="font-size:11.5px;font-weight:500;color:var(--muted)">${f.note}</span></span>${on ? ICON.check : ""}</button>`;
      }).join("")}</div>
      <div class="eyebrow">PROCESSING</div>
      <div class="chips">${PROCESSORS.map((f) => {
        const on = state.processor === f.id;
        return `<button class="chip-btn ${on ? "on" : ""}" style="${on ? "border-color:rgba(255,255,255,.45);background:rgba(255,255,255,.06);" : ""}" data-action="processor" data-id="${f.id}"><span class="nm">${f.name}<br><span style="font-size:11.5px;font-weight:500;color:var(--muted)">${f.note}</span></span>${on ? ICON.check : ""}</button>`;
      }).join("")}</div>
      <div class="eyebrow">SONG SECTIONS</div>
      <button class="check-row" data-action="auto-sections" aria-pressed="${state.autoSections}"><span class="check${state.autoSections ? " on" : ""}">${state.autoSections ? ICON.check : ""}</span>Find the sections (intro, verse, chorus…)</button>
      <button class="cta" style="margin-top:22px" data-action="split">${ICON.scissors}Extract stems</button>
      ${modelNote() /* Android */}
      ${extractProgressCard()}
    </div>
  </div>`;
}

function refreshModels() {
  fetch("/api/config").then((r) => r.json()).then((c) => { if (c && c.models) state.models = c.models; }).catch(() => {});
}

// Android: what the next song will download first, once.
function modelNote() {
  const m = state.models;
  if (!m) return "";
  const need = [];
  if (!m.separation) need.push("the separation model (55 MB)");
  if (!m.beat) need.push("the beat model for the click track (82 MB)");
  if (state.autoSections && !m.sections) need.push("the section models (27 MB)");
  if (state.selected.vocals && state.vocalSplitMode === "split" && !m.vocal_split) need.push("the vocal split model (53 MB)");
  if (!need.length) return "";
  return `<div class="model-note">The first song downloads ${need.join(" and ")}, once.</div>`;
}

// Submit an extraction (URL or file) and follow it to completion.
async function startExtraction() {
  const stems = Object.keys(state.selected).filter((k) => state.selected[k]);
  if (!stems.length) { toast("Pick at least one stem."); return; }
  const file = state.extractFile;
  const url = (state.extractUrl || "").trim();
  if (!file && !url) { toast("Choose an audio file first."); return; }

  let init, title;
  if (file) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("stems", JSON.stringify(stems));
    fd.append("format", state.stemFormat); // Android
    init = { method: "POST", body: fd };
    title = file.name;
  } else {
    init = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url, stems }) };
    title = url;
  }

  const vocalSplitMode = state.selected.vocals ? state.vocalSplitMode : "all";
  state.extractJob = { id: null, title, status: "queued", progress: 0, stage: "Queued", vocalSplitMode };
  render();

  let jobId;
  try {
    const res = await fetch("/api/jobs", init);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    jobId = data.job_id;
  } catch (e) {
    console.warn("[mobile] extraction submit failed:", e);
    state.extractJob = null;
    toast(`Couldn't start: ${e.message}`);
    render();
    return;
  }

  state.extractJob.id = jobId;
  if (state.autoSections) saveAutoSections(false); // Android: the choice was for this import (desktop resets it too)
  state.extractFile = null;
  state.extractUrl = "";
  state.vocalSplitMode = "all";
  followExtraction(jobId);
  render();
}

function _finishExtractJob(jobId) {
  const finishedId = jobId;
  setTimeout(() => {
    if (state.extractJob && state.extractJob.id === finishedId) {
      state.extractJob = null;
      if (state.tab === "extract") render();
    }
  }, 4000);
}

// Chains the on-demand lead/backing vocal split (#275) onto the base job's
// completion when the user picked "Lead + Backing" on the Extract screen --
// from their perspective this is one action, even though the backend keeps
// it a separate, best-effort pass over the already-done job.
async function _runVocalSplitThenFinish(jobId) {
  state.extractJob.stage = "Splitting lead/backing vocals…";
  if (state.tab === "extract") render();
  try {
    const res = await fetch(`/api/jobs/${jobId}/vocal-split`, { method: "POST" });
    if (!res.ok && res.status !== 202) throw new Error(String(res.status));
    toast("Stems ready!");
  } catch (e) {
    console.warn("[mobile] vocal split failed:", e);
    // The base separation still succeeded -- the split is a best-effort
    // extra, so this is a lesser notice, not a failure of the whole import.
    toast("Stems ready (lead/backing split didn't work this time).");
  }
  loadLibrary();
  _finishExtractJob(jobId);
}

function _onExtractState(jobId, s) {
  if (!state.extractJob || state.extractJob.id !== jobId) return false;
  state.extractJob.status = s.status;
  state.extractJob.progress = s.progress || 0;
  state.extractJob.stage = s.stage || s.status;
  if (state.tab === "extract") render();
  if (["done", "error", "cancelled", "stopped"].includes(s.status)) {
    if (s.status === "done" && state.extractJob.vocalSplitMode === "split") {
      _runVocalSplitThenFinish(jobId);
    } else {
      if (s.status === "done") { toast("Stems ready!"); loadLibrary(); }
      else if (s.status === "error") toast("Extraction failed.");
      _finishExtractJob(jobId);
    }
    refreshModels(); // Android: the first song may have downloaded the models
    return true; // terminal for the base-job follow (SSE/poll can stop)
  }
  return false;
}

// SSE with a REST-poll fallback (mirrors the desktop's resilience, simplified).
function followExtraction(jobId) {
  if (extractES) { extractES.close(); extractES = null; }
  if (extractPoll) { clearInterval(extractPoll); extractPoll = null; }

  const startPolling = () => {
    if (extractPoll) return;
    extractPoll = setInterval(async () => {
      try {
        const r = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        if (!r.ok) throw new Error(String(r.status));
        if (_onExtractState(jobId, await r.json())) { clearInterval(extractPoll); extractPoll = null; }
      } catch { /* keep polling */ }
    }, 2500);
  };

  try {
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    extractES = es;
    es.onmessage = (ev) => {
      let s;
      try { s = JSON.parse(ev.data); } catch { return; }
      if (_onExtractState(jobId, s)) { es.close(); extractES = null; }
    };
    es.onerror = () => { es.close(); extractES = null; startPolling(); };
  } catch {
    startPolling();
  }
}

function miniPlayer() {
  if (state.tab === "mixer" || !state.current) return "";
  const c = state.current;
  return `<div class="mini" data-action="tab" data-tab="mixer">
    <div class="mini-art" style="${artStyle(c)}">${artLabel(c)}</div>
    <div class="mini-info"><div class="t">${esc(c.title)}</div><div class="s">${esc(c.sub)}</div></div>
    <button class="mini-play" data-action="play-mini">${state.playing ? ICON.pause(17, "#1a1206") : ICON.play(18, "#1a1206")}</button>
  </div>`;
}

function tabBar() {
  const t = (tab, icon, label) => `<button class="tab ${state.tab === tab ? "on" : ""}" data-action="tab" data-tab="${tab}">${icon}${label}</button>`;
  return `<div class="tabbar">${t("extract", ICON.tabExt, "Extract")}${t("mixer", ICON.tabMix, "Mixer")}${t("library", ICON.tabLib, "Library")}</div>`;
}

function render() {
  let screen = mixerScreen();
  if (state.tab === "library") screen = libraryScreen();
  else if (state.tab === "extract") screen = extractScreen();
  const mini = miniPlayer();
  // Android: the screens leave room for the mini player too while it shows,
  // so it does not cover the last item (e.g. the Extract stems button).
  app.classList.toggle("has-mini", mini !== "");
  app.innerHTML = screen + mini + tabBar();
  _lastPlayedBar = -1; // bars were just rebuilt; force a repaint on next tick
  wireFaders();
  wireSwipe();
}

// Suppresses the synthetic click that follows a swipe gesture, so swiping a
// row open doesn't immediately count as a tap.
let suppressClick = false;

// Swipe a library row left to reveal its Delete button (iOS-style). Wired
// imperatively after each render.
function wireSwipe() {
  app.querySelectorAll(".track-wrap .track").forEach((row) => {
    const id = row.dataset.id;
    let x0 = 0, y0 = 0, active = false, horizontal = false;
    const base = () => (state.swipedTrackId === id ? -84 : 0);

    row.addEventListener("pointerdown", (e) => {
      x0 = e.clientX; y0 = e.clientY; active = true; horizontal = false;
      row.style.transition = "none";
    });
    row.addEventListener("pointermove", (e) => {
      if (!active) return;
      const dx = e.clientX - x0;
      const dy = e.clientY - y0;
      if (!horizontal && Math.abs(dx) > 8 && Math.abs(dx) > Math.abs(dy)) horizontal = true;
      if (horizontal) {
        e.preventDefault();
        const tx = Math.max(-84, Math.min(0, base() + dx));
        row.style.transform = `translateX(${tx}px)`;
      }
    });
    const end = (e) => {
      if (!active) return;
      active = false;
      if (!horizontal) { row.style.transition = ""; row.style.transform = ""; return; }
      suppressClick = true;
      setTimeout(() => { suppressClick = false; }, 60);
      const dx = e.clientX - x0;
      const open = dx < -40 ? true : dx > 40 ? false : state.swipedTrackId === id;
      // Animate to the snap position by toggling the class (no DOM rebuild, so
      // it transitions smoothly from wherever the finger let go).
      closeOtherSwipes(row.closest(".track-wrap"));
      row.style.transition = "transform 0.22s cubic-bezier(0.22,1,0.36,1)";
      row.closest(".track-wrap").classList.toggle("swiped", open);
      row.style.transform = "";
      state.swipedTrackId = open ? id : null;
    };
    row.addEventListener("pointerup", end);
    row.addEventListener("pointercancel", end);
  });
}

// Smoothly close any revealed row (optionally except `keep`), without a full
// re-render so the close animates.
function closeOtherSwipes(keep) {
  app.querySelectorAll(".track-wrap.swiped").forEach((w) => {
    if (w === keep) return;
    const r = w.querySelector(".track");
    if (r) { r.style.transition = "transform 0.22s cubic-bezier(0.22,1,0.36,1)"; r.style.transform = ""; }
    w.classList.remove("swiped");
  });
}

function closeSwipe() {
  state.swipedTrackId = null;
  closeOtherSwipes(null);
}

// The same flag the desktop's hearts set (#734). The row changes at once and
// goes back if the server refuses, so a tap never shows a state that was not
// kept.
async function toggleFavorite(id) {
  const track = state.tracks.find((t) => t.id === id);
  if (!track) return;
  track.favorite = !track.favorite;
  render();
  try {
    const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/favorite`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ favorite: track.favorite }),
    });
    if (!res.ok) throw new Error(`PUT favorite -> ${res.status}`);
  } catch (e) {
    console.warn("[mobile] could not save the favourite:", e);
    track.favorite = !track.favorite;
    toast("Could not save the favourite");
    render();
  }
}

async function deleteTrack(id) {
  state.swipedTrackId = null;
  try {
    // Trash, not DELETE. This is a swipe on a touch screen with no undo, and
    // DELETE removes the stems from disk for good. The desktop has always
    // moved tracks to a Trash folder instead; now that the Trash lives on the
    // server, this screen can do the same thing rather than being the one
    // place in StemDeck where a stray thumb destroys a separation.
    await fetch(`/api/jobs/${encodeURIComponent(id)}/trash`, { method: "POST" });
  } catch (e) {
    console.warn("[mobile] could not move track to trash:", e);
  }
  state.tracks = state.tracks.filter((t) => t.id !== id);
  if (!state.tracks.length) state.libState = "empty";
  if (state.current?.id === id) {
    teardownClick(); // Android
    if (engine) { engine.destroy(); engine = null; engineTrackId = null; engineReady = false; }
    state.current = null;
    state.playing = false;
  }
  toast("Moved to Trash");
  render();
}

// Faders + waveform seek need live pointer drag, so they're wired imperatively
// after each render (the rest of the UI uses click delegation below).
function wireFaders() {
  // Android: section blocks. A tap plays from the section's start; holding
  // one for half a second loops it (and holding it again stops the loop).
  app.querySelectorAll("[data-sec]").forEach((el) => {
    el.addEventListener("pointerdown", (e) => {
      const list = currentSections();
      const sec = list && list.find((x) => x.id === el.dataset.sec);
      if (!sec || !engine) return;
      e.preventDefault();
      const x0 = e.clientX, y0 = e.clientY;
      let held = false, gone = false;
      const timer = setTimeout(() => { held = true; navigator.vibrate?.(15); setSectionLoop(sec); }, 500);
      const mv = (ev) => { if (Math.abs(ev.clientX - x0) > 10 || Math.abs(ev.clientY - y0) > 10) { gone = true; clearTimeout(timer); } };
      const up = () => {
        clearTimeout(timer);
        el.removeEventListener("pointermove", mv);
        el.removeEventListener("pointerup", up);
        el.removeEventListener("pointercancel", up);
        if (held || gone) return;
        const dur = curDuration();
        // Outside a looped section the loop would pull playback straight back.
        if (state.loopSection && state.loopSection !== sec.id) { state.loopSection = null; engine.setLoop(false, 0, 0); }
        engine.seek(sec.start);
        onEngineTime(sec.start);
        if (dur) render();
      };
      el.addEventListener("pointermove", mv);
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
    });
  });

  app.querySelectorAll("[data-fader]").forEach((el) => {
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const id = el.dataset.id;
      const fill = el.querySelector(".fader-fill");
      const knob = el.querySelector(".fader-knob");
      const rect = el.getBoundingClientRect();
      const set = (cx) => {
        let v = (cx - rect.left) / rect.width;
        v = Math.max(0, Math.min(1, v));
        state.vols[id] = v;
        const pct = Math.round(v * 100) + "%";
        fill.style.width = pct;
        knob.style.left = pct;
        if (engine && laneActive(id)) engine.setGain(id, v);
      };
      set(e.clientX);
      el.setPointerCapture(e.pointerId);
      const mv = (ev) => set(ev.clientX);
      const up = () => {
        el.removeEventListener("pointermove", mv);
        el.removeEventListener("pointerup", up);
      };
      el.addEventListener("pointermove", mv);
      el.addEventListener("pointerup", up);
    });
  });

  const speedSlider = app.querySelector("[data-speed]");
  if (speedSlider) {
    speedSlider.addEventListener("input", () => {
      const rate = parseFloat(speedSlider.value);
      state.speed = rate;
      const valEl = speedSlider.parentElement?.querySelector(".speed-row-val");
      if (valEl) valEl.textContent = `${rate % 1 === 0 ? rate.toFixed(1) : rate}x`;
      if (engine) engine.setPlaybackRate(rate);
    });
  }

  // Android: click volume, accent and count-in.
  const clickVol = app.querySelector("[data-click-vol]");
  if (clickVol) {
    clickVol.addEventListener("input", () => {
      state.click.volume = Math.max(0, Math.min(1, parseFloat(clickVol.value)));
      const valEl = clickVol.parentElement?.querySelector(".speed-row-val");
      if (valEl) valEl.textContent = `${Math.round(state.click.volume * 100)}%`;
      metronome?.setVolume(state.click.volume);
    });
    clickVol.addEventListener("change", saveClickSettings);
  }
  app.querySelector("[data-click-accent]")?.addEventListener("change", (e) => {
    state.click.accent = parseInt(e.target.value, 10);
    applyClickAccent();
    saveClickSettings();
  });
  app.querySelector("[data-click-countin]")?.addEventListener("change", (e) => {
    state.click.countIn = parseInt(e.target.value, 10) || 0;
    saveClickSettings();
  });

  app.querySelectorAll("[data-key-step]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!engine?.setStemPitch) return;
      const next = clampPitch(state.pitch + Number(btn.dataset.keyStep));
      if (next === state.pitch) return;
      state.pitch = next;
      // Every lane together: this screen has no per-lane keys to preserve.
      for (const lane of lanes()) engine.setStemPitch(lane.name, next);
      render();
    });
  });

  const bars = app.querySelector("[data-seek]");
  if (bars) {
    bars.addEventListener("pointerdown", (e) => {
      if (!engine || !curDuration()) return;
      e.preventDefault();
      const rect = bars.getBoundingClientRect();
      const head = bars.querySelector(".playhead");
      const seek = (cx) => {
        const frac = Math.max(0, Math.min(1, (cx - rect.left) / rect.width));
        if (head) head.style.left = frac * 100 + "%";
        const cur = app.querySelector(".wave-times .cur");
        if (cur) cur.textContent = fmt(frac * curDuration());
        state.progress = frac;
        paintWaveProgress();
        seekToFraction(frac);
      };
      seek(e.clientX);
      bars.setPointerCapture(e.pointerId);
      const mv = (ev) => seek(ev.clientX);
      const up = () => {
        bars.removeEventListener("pointermove", mv);
        bars.removeEventListener("pointerup", up);
      };
      bars.addEventListener("pointermove", mv);
      bars.addEventListener("pointerup", up);
    });
  }
}

app.addEventListener("click", (e) => {
  if (suppressClick) { suppressClick = false; return; }
  const t = e.target.closest("[data-action]");
  if (!t) return;
  ensureAudioCtx(); // any in-app tap unlocks audio on mobile
  const a = t.dataset.action;
  // A row is swiped open: any tap other than its Delete just closes it.
  if (state.swipedTrackId && a !== "delete") {
    closeSwipe();
    return;
  }
  switch (a) {
    case "delete":
      deleteTrack(t.dataset.id);
      return;
    case "favorite":
      // The row underneath opens the track; the heart is its own button.
      e.stopPropagation();
      toggleFavorite(t.dataset.id);
      return;
    case "tab":
      state.tab = t.dataset.tab;
      break;
    case "mixview":
      state.mixerView = t.dataset.view;
      break;
    case "prev":
      prevTrack();
      return;
    case "next":
      nextTrack();
      return;
    case "play":
    case "play-mini":
      e.stopPropagation();
      togglePlay();
      return;
    case "mute":
      state.muted[t.dataset.id] = !state.muted[t.dataset.id];
      applyMix();
      break;
    case "solo":
      state.solo[t.dataset.id] = !state.solo[t.dataset.id];
      applyMix();
      break;
    case "auto-sections": // Android
      saveAutoSections(!state.autoSections);
      break;
    case "click-toggle": // Android
      state.click.on = !state.click.on;
      metronome?.setEnabled(state.click.on);
      saveClickSettings();
      break;
    case "click-mult": // Android
      state.click.mult = Number(t.dataset.mult);
      metronome?.setMultiplier(state.click.mult);
      break;
    case "export-click": // Android
      state.exportClick = !state.exportClick;
      break;
    case "export-zip": // Android
      exportStemsZip();
      return;
    case "processor": // Android
      state.processor = t.dataset.id;
      fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ gpu: state.processor === "gpu" }) })
        .catch(() => toast("Couldn't save the setting."));
      break;
    case "format": // Android
      state.stemFormat = t.dataset.id;
      fetch("/api/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ stem_format: state.stemFormat }) })
        .catch(() => toast("Couldn't save the format."));
      break;
    case "chip":
      state.selected[t.dataset.id] = !state.selected[t.dataset.id];
      // Vocals switched on from its own chip has not been told which of the
      // two ways it should come out, so it takes Combined, as on the desktop.
      if (t.dataset.id === "vocals" && state.selected.vocals) state.vocalSplitMode = "all";
      break;
    case "vocalmode":
      // The pair is the vocals control, not a setting that waits for the chip.
      // Pressing the mode already in force switches vocals off; pressing the
      // other switches mode and leaves them on; either one from cold switches
      // them on in that mode.
      if (state.selected.vocals && state.vocalSplitMode === t.dataset.mode) {
        state.selected.vocals = false;
      } else {
        state.vocalSplitMode = t.dataset.mode;
        state.selected.vocals = true;
      }
      break;
    case "qual":
      state.quality = t.dataset.q;
      break;
    case "filter":
      state.filter = t.dataset.filter;
      break;
    case "export":
      exportMix(t.dataset.kind || "wav");
      return;
    case "split":
      startExtraction();
      return;
    case "pick-file":
      document.getElementById("ext-file")?.click();
      return;
    case "open": {
      const track = state.tracks.find((x) => x.id === t.dataset.id);
      if (track) openTrack(track);
      return;
    }
    case "reimport": {
      const track = state.tracks.find((x) => x.id === t.dataset.id);
      if (track) reimportTrack(track);
      return;
    }
    case "reload":
      loadLibrary();
      return;
    default:
      return;
  }
  render();
});

// URL field is uncontrolled-but-tracked: store keystrokes in state without
// re-rendering (which would steal focus), so the value survives later renders.
app.addEventListener("input", (e) => {
  if (e.target.id === "ext-url") state.extractUrl = e.target.value;
});
app.addEventListener("change", (e) => {
  if (e.target.id === "ext-file") {
    state.extractFile = e.target.files?.[0] || null;
    render();
  }
});

// A track's files are gone (folder deleted or moved outside the app).
// URL-sourced tracks can be rebuilt by re-running the import; a local upload
// has no source bytes left (the pipeline deletes the upload once it's done
// with it), so that just gets a toast pointing at re-upload instead.
async function reimportTrack(card) {
  if (!card.sourceUrl || card.sourceUrl.startsWith("local:")) {
    toast("This track's audio is gone. Re-upload it to restore it.");
    return;
  }
  toast("Reimporting…");
  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: card.sourceUrl, stems: card.selectedStems || [] }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    await loadLibrary();
  } catch (e) {
    console.warn("[mobile] reimport failed:", e);
    toast(`Couldn't reimport: ${e.message}`);
  }
}

// Load the real library from /api/jobs (newest first). On success, seed the
// Mixer with the most recent track if nothing is selected yet.
async function loadLibrary() {
  state.libState = "loading";
  render();
  try {
    const jobs = await fetchJobs();
    state.tracks = jobs.map(jobToCard).sort((a, b) => b.createdAt - a.createdAt);
    state.libState = state.tracks.length ? "ready" : "empty";
  } catch (e) {
    console.warn("[mobile] failed to load library:", e);
    state.libState = "error";
  }
  render();
}

// Android: refresh the library quietly while something is being separated
// (an import shared from another app, or one started before a reload), so
// its dot turns green without a manual reload.
async function refreshLibraryQuietly() {
  try {
    const jobs = await fetchJobs();
    state.tracks = jobs.map(jobToCard).sort((a, b) => b.createdAt - a.createdAt);
    state.libState = state.tracks.length ? "ready" : "empty";
    if (state.tab === "library" && !state.swipedTrackId) render();
  } catch { /* the next tick tries again */ }
}
setInterval(() => {
  if (document.hidden) return;
  if (state.tracks.some((t) => t.status === "processing")) refreshLibraryQuietly();
}, 3000);
window.addEventListener("hashchange", () => {
  const tab = location.hash.slice(1);
  if (["library", "extract", "mixer"].includes(tab)) { state.tab = tab; refreshLibraryQuietly(); render(); }
});

render();
loadLibrary();
