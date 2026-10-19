#!/usr/bin/env python3
"""Playwright checks of the mobile UI in app/src/main/assets/web against a
fake of the app's local server (the API shape LocalServer.kt serves).

  python3 scripts/ui_test.py [--shots DIR]

Runs at the phone sizes the app is used at (384 x 640, 740 and 832) and
checks what earlier releases got wrong or what this one adds: the Extract
button stays clear of the mini player, the click track's controls appear
for a song with a beat grid and drive the scheduler, the beat strip matches
the grid, and the Analysis exports ask for the right files.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "app" / "src" / "main" / "assets" / "web"
SR = 44100
SECONDS = 24


def wav_bytes(freq: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for i in range(SR * SECONDS):
            v = int(3000 * math.sin(2 * math.pi * freq * i / SR))
            frames += v.to_bytes(2, "little", signed=True) * 2
        w.writeframes(bytes(frames))
    return buf.getvalue()


STEMS = {n: wav_bytes(f) for n, f in [("drums", 110), ("bass", 55), ("vocals", 330)]}
BEATS = [round(0.5 + i * 0.5, 6) for i in range(46)]
GRID = {
    "version": 1, "source": "drums", "detector": "beat_this", "bars": [{"beat": 0, "beats_per_bar": 4}],
    "bpm": 120.0, "duration": float(SECONDS), "confidence": 91, "interval_cv": 0.01, "irregularity": 0.0,
    "refined": 46, "corrected": 0, "onsets": [], "detected": 46, "extrapolated_head": 0, "extrapolated_tail": 0,
    "gap_filled": 0, "phase_shifted": False, "beats": BEATS,
}


SECTIONS = [
    {"id": "auto-001", "name": "Intro", "kind": "intro", "start": 0.0, "end": 8.0, "color": "#4a7fff"},
    {"id": "auto-002", "name": "Chorus", "kind": "chorus", "start": 8.0, "end": 14.0, "color": "#9a4aff"},
    {"id": "auto-003", "name": "Chorus", "kind": "chorus", "start": 14.0, "end": 24.0, "color": "#9a4aff"},
]


def job(jid: str, title: str, with_grid: bool) -> dict:
    return {
        "job_id": jid, "status": "done", "progress": 1.0, "stage": "Done", "title": title,
        "duration": float(SECONDS), "thumbnail": None, "bpm": 120 if with_grid else None,
        "key": "A min" if with_grid else None, "scale": "Natural Minor" if with_grid else None,
        "key_confidence": 80 if with_grid else None, "lufs": -9.0, "peak_db": -0.3, "dynamic_range": 8.7,
        "tempo_stability": 99 if with_grid else None, "stem_presence": {"drums": 100, "bass": 70, "vocals": 40},
        "stems": [{"name": n, "url": f"/api/jobs/{jid}/stems/{n}.wav"} for n in STEMS],
        "selected_stems": list(STEMS), "source_url": f"local:{title}", "source_format": None,
        "has_video": False, "has_lyrics": False, "error": None, "compute_device": "test",
        "trashed_at": None, "favorite": False, "created_at": 2.0 if with_grid else 1.0, "stem_format": "wav",
        "analysis_notes": {} if with_grid else {"beats": "No beat grid: only 3 beats detected."},
        "sections": SECTIONS if with_grid else None, "sections_source": "automatic" if with_grid else None,
    }


LRC = "\n".join(f"[00:{t:02d}.00]Line {i} of the song" for i, t in enumerate(range(1, 24, 2)))
LYRICS = {"v": 1, "source": "lrclib", "track": "Grid Song", "artist": "Test", "album": "", "duration": 24.0,
          "synced": LRC, "plain": "", "instrumental": False, "lrclib_id": 7, "timing": "exact",
          "others": [{"v": 1, "source": "lrclib", "track": "Grid Song", "artist": "Test", "album": "Live", "duration": 31.0,
                      "synced": "", "plain": "Other words", "instrumental": False, "lrclib_id": 8}]}
JOBS = [job("a1b2c3d4e5f6", "Grid Song", True), job("b1b2c3d4e5f6", "Old Song", False)]
JOBS[1]["analysis_notes"]["lyrics"] = "No lyrics: LRCLIB has none."
JOBS[1]["audio_tags"] = {"title": "Old Song", "artist": "Old Band"}
# A song after the lead/backing split: those two lanes replace Vocals.
_split = job("c1b2c3d4e5f6", "Split Song", False)
_split["stems"] += [{"name": n, "url": f"/api/jobs/c1b2c3d4e5f6/stems/vocals.wav"} for n in ("lead_vocals", "backing_vocals")]
_split["vocal_split"] = "done"
_split["created_at"] = 0.5
JOBS.append(_split)


class Api(BaseHTTPRequestHandler):
    # auto_sections left on by an "earlier session": the page must switch it off.
    settings = {"stem_format": "mp3", "gpu": True, "click": {}, "auto_sections": True, "lyrics_lookup": False}
    lookups: list = []
    offsets: list = []
    puts: list = []
    downloads: list = []

    def log_message(self, *a):  # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        m = re.match(r"^/api/jobs/([0-9a-f]{12})/lyrics/(lookup|offset)$", self.path)
        if m and m.group(2) == "lookup":
            Api.lookups.append(body)
            if not Api.settings.get("lyrics_lookup"):
                return self.send(403, b'{"detail":"off","disabled":true}', "application/json")
            return self.send(200, json.dumps(LYRICS).encode(), "application/json")
        if m and m.group(2) == "offset":
            Api.offsets.append(body.get("offset_sec"))
            return self.send(200, json.dumps({"offset_sec": body.get("offset_sec")}).encode(), "application/json")
        if self.path == "/api/settings":
            Api.puts.append(body)
            Api.settings.update({k: v for k, v in body.items() if k != "click"})
            if "click" in body:
                Api.settings["click"] = body["click"]
            return self.send(200, json.dumps(Api.settings).encode(), "application/json")
        self.send(404, b"{}", "application/json")

    do_POST = do_PUT

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/mobile":
            return self.send(302, b"", "text/plain", {"Location": "/mobile/"})
        if path == "/api/config":
            return self.send(200, json.dumps({"platform": "android", "models": {"separation": True, "beat": False, "sections": False}}).encode(), "application/json")
        if path == "/api/settings":
            return self.send(200, json.dumps(Api.settings).encode(), "application/json")
        if path == "/api/jobs":
            return self.send(200, json.dumps(JOBS).encode(), "application/json")
        m = re.match(r"^/api/jobs/([0-9a-f]{12})(/.*)?$", path)
        if m:
            j = next((x for x in JOBS if x["job_id"] == m.group(1)), None)
            rest = m.group(2) or ""
            if j is None:
                return self.send(404, b'{"detail":"job not found"}', "application/json")
            if rest == "":
                return self.send(200, json.dumps(j).encode(), "application/json")
            if rest == "/lyrics":
                if j["job_id"] == "a1b2c3d4e5f6":
                    return self.send(200, json.dumps(LYRICS).encode(), "application/json")
                return self.send(404, b'{"detail":"no lyrics","others":[]}', "application/json")
            if rest == "/stems/beats.json":
                if j["bpm"] is None:
                    return self.send(404, b'{"detail":"beat grid not found"}', "application/json")
                return self.send(200, json.dumps(GRID).encode(), "application/json")
            if rest in ("/stems/all.zip", "/mixdown.wav"):
                Api.downloads.append(self.path)
                return self.send(200, b"PK", "application/zip" if rest.endswith("zip") else "audio/wav",
                                 {"Content-Disposition": "attachment; filename=x"})
            sm = re.match(r"^/stems/(\w+)\.wav$", rest)
            if sm and sm.group(1) in STEMS:
                data = STEMS[sm.group(1)]
                rng = self.headers.get("Range")
                if rng:
                    a, _, b = rng.removeprefix("bytes=").partition("-")
                    a = int(a)
                    b = int(b) if b else len(data) - 1
                    b = min(b, len(data) - 1)
                    return self.send(206, data[a:b + 1], "audio/wav",
                                     {"Content-Range": f"bytes {a}-{b}/{len(data)}", "Accept-Ranges": "bytes"})
                return self.send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})
            return self.send(404, b"{}", "application/json")
        f = (WEB / path.lstrip("/")).resolve()
        if path.endswith("/"):
            f = f / "index.html"
        if WEB in f.parents and f.is_file():
            ct = {"html": "text/html", "js": "text/javascript", "css": "text/css", "json": "application/json", "ttf": "font/ttf"}.get(f.suffix[1:], "application/octet-stream")
            return self.send(200, f.read_bytes(), ct)
        self.send(404, b"not found", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    server = ThreadingHTTPServer(("127.0.0.1", 0), Api)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failures: list[str] = []

    def check(cond: bool, what: str):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            failures.append(what)

    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for h in (640, 740, 832):
            page = browser.new_page(viewport={"width": 384, "height": h}, device_scale_factor=2)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            outside: list[str] = []  # the app promises to contact nobody unasked
            page.on("request", lambda r: outside.append(r.url) if not r.url.startswith(base) and not r.url.startswith("data:") else None)
            # Count the clicks the scheduler makes (one oscillator each).
            page.add_init_script("""
              window.__osc = 0;
              const orig = AudioContext.prototype.createOscillator;
              AudioContext.prototype.createOscillator = function () { window.__osc++; return orig.call(this); };
            """)
            Api.settings["lyrics_lookup"] = False  # each size starts with the lookup off
            page.goto(base + "/mobile/#library")
            page.wait_for_selector(".track[data-id='a1b2c3d4e5f6']")
            page.click(".track-load[data-id='a1b2c3d4e5f6']")
            page.wait_for_selector(".t-play:not([disabled])", timeout=15000)
            page.wait_for_selector(".click-row")
            check(page.locator(".beat-strip").count() == 1, f"{h}: beat strip shown for a song with a grid")
            d = page.get_attribute(".beat-strip .bs-beat", "d") or ""
            check(d.count("M") == len(BEATS), f"{h}: one tick per beat ({d.count('M')})")
            bars = page.get_attribute(".beat-strip .bs-bar", "d") or ""
            check(bars.count("M") == math.ceil(len(BEATS) / 4), f"{h}: one bar mark every 4 beats ({bars.count('M')})")
            # Sections: one block each, desktop's names (numbered when repeated).
            check(page.locator(".sec-block").count() == len(SECTIONS), f"{h}: one block per section")
            names = [page.inner_text(f".sec-block[data-sec='{x['id']}']").strip() for x in SECTIONS]
            check(names == ["Intro", "Chorus 1", "Chorus 2"], f"{h}: section names {names}")
            page.click(".sec-block[data-sec='auto-002']")
            page.wait_for_timeout(200)
            check(page.inner_text(".wave-times .cur").strip() == "0:08", f"{h}: tapping a section seeks to its start ({page.inner_text('.wave-times .cur')})")
            box = page.locator(".sec-block[data-sec='auto-003']").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down(); page.wait_for_timeout(700); page.mouse.up()
            check(page.locator(".sec-block.looped[data-sec='auto-003']").count() == 1, f"{h}: holding a section loops it")
            check(page.inner_text(".wave-times .cur").strip() == "0:14", f"{h}: the loop starts at the section ({page.inner_text('.wave-times .cur')})")
            box = page.locator(".sec-block[data-sec='auto-003']").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down(); page.wait_for_timeout(700); page.mouse.up()
            check(page.locator(".sec-block.looped").count() == 0, f"{h}: holding it again ends the loop")
            if shots:
                page.screenshot(path=str(shots / f"mixer-{h}.png"), full_page=True)
            # Click on, then play: the scheduler makes oscillators.
            page.click("[data-action='click-toggle']")
            page.wait_for_timeout(300)
            check(page.get_attribute("[data-action='click-toggle']", "aria-pressed") == "true", f"{h}: click toggles on")
            check(any(x.get("click", {}).get("on") is True for x in Api.puts), f"{h}: click setting saved to the app")
            page.click(".t-play")
            page.wait_for_timeout(1800)
            n1 = page.evaluate("window.__osc")
            check(n1 >= 2, f"{h}: clicks scheduled while playing ({n1})")
            page.select_option("[data-click-accent]", "3")
            page.wait_for_timeout(300)  # the save is a fetch
            check(any(x.get("click", {}).get("accent") == 3 for x in Api.puts), f"{h}: accent saved")
            page.click("[data-action='click-mult'][data-mult='2']")
            check(page.get_attribute("[data-action='click-mult'][data-mult='2']", "aria-checked") == "true", f"{h}: double time selected")
            page.click(".t-play")  # pause
            page.select_option("[data-click-countin]", "1")
            # Count-in: with the click off, playing still sounds the count.
            page.click("[data-action='click-toggle']")
            before = page.evaluate("window.__osc")
            page.click(".t-play")
            page.wait_for_timeout(600)
            after = page.evaluate("window.__osc")
            check(after - before >= 3, f"{h}: a 1-bar count-in with the click off ({after - before} clicks)")
            page.click(".t-play")
            # Analysis tab: beat grid and exports.
            page.click("[data-action='mixview'][data-view='analysis']")
            check("120.0 BPM" in page.inner_text(".grid-info"), f"{h}: beat grid summary")
            Api.downloads.clear()
            page.click("[data-action='mute'][data-id='bass']") if page.locator("[data-action='mute'][data-id='bass']").count() else None
            page.click("[data-action='export-zip']")
            page.wait_for_timeout(300)
            check(any("/stems/all.zip?stems=drums%2Cbass%2Cvocals" in u for u in Api.downloads), f"{h}: ZIP asks for the active stems ({Api.downloads})")
            page.click("[data-action='export-click']")
            page.click("[data-action='export'][data-kind='wav']")
            page.wait_for_timeout(300)
            check(any("click=1" in u and "click_accent=3" in u and "click_mult=2" in u for u in Api.downloads), f"{h}: mix export carries the click options")
            if shots:
                page.screenshot(path=str(shots / f"analysis-{h}.png"), full_page=True)
            # Lyrics: lines from the LRC, the current one follows the playhead, a tap seeks.
            page.click("[data-action='mixview'][data-view='lyrics']")
            page.wait_for_selector(".ly-line")
            check(page.locator(".ly-line").count() == LRC.count("\n") + 1, f"{h}: one row per LRC line")
            page.click(".ly-line[data-ly='2']")
            page.wait_for_timeout(300)
            check(page.get_attribute(".ly-line.cur", "data-ly") == "2", f"{h}: tapping a line highlights it")
            check(page.inner_text(".wave-times .cur").strip() == "0:05", f"{h}: tapping a line seeks to it ({page.inner_text('.wave-times .cur')})")
            page.click(".t-play")
            page.wait_for_timeout(3000)
            cur = page.get_attribute(".ly-line.cur", "data-ly")
            shown = page.inner_text(".wave-times .cur").strip()
            check(cur == "3", f"{h}: the highlight follows the playhead (line {cur} at {shown})")
            wiped = page.evaluate("getComputedStyle(document.querySelector('.ly-line.cur .ly-w')).getPropertyValue('--p')")
            check(bool(wiped.strip()), f"{h}: the current line's words are wiped ({wiped!r})")
            page.click(".t-play")
            page.click("[data-action='ly-nudge'][data-d='0.5']")
            page.wait_for_timeout(700)
            check(0.5 in Api.offsets, f"{h}: the timing offset is saved ({Api.offsets})")
            check("Live" in page.inner_text(".ly-pad") or page.locator(".ly-other").count() == 1, f"{h}: other versions offered")
            if shots:
                page.screenshot(path=str(shots / f"lyrics-{h}.png"), full_page=False)
            page.click("[data-action='mixview'][data-view='analysis']")
            # A song with no grid: no click controls, and the reason shown.
            page.click("[data-action='mixview'][data-view='stems']")
            page.click("[data-action='tab'][data-tab='library']")
            page.click(".track-load[data-id='b1b2c3d4e5f6']")
            page.wait_for_selector(".t-play:not([disabled])", timeout=15000)
            check(page.locator(".click-row").count() == 0, f"{h}: no click controls without a grid")
            check("only 3 beats" in page.inner_text(".click-note"), f"{h}: says why there is no grid")
            # No lyrics, lookup off: the search form, and Settings to turn it on.
            page.click("[data-action='mixview'][data-view='lyrics']")
            page.wait_for_selector("[data-ly-title]")
            check(page.input_value("[data-ly-title]") == "Old Song" and page.input_value("[data-ly-artist]") == "Old Band", f"{h}: search prefilled from the tags")
            check("lookup is off" in page.inner_text(".ly-pad"), f"{h}: says the lookup is off")
            page.click(".ly-pad [data-action='tab'][data-tab='settings']")
            page.wait_for_selector(".set-row")
            check(page.get_attribute("[data-action='set-lyrics']", "aria-pressed") == "false", f"{h}: lyrics lookup off by default")
            page.click("[data-action='set-lyrics']")
            page.wait_for_timeout(300)
            check(Api.settings.get("lyrics_lookup") is True, f"{h}: lyrics lookup turned on in Settings")
            page.click("[data-action='tab'][data-tab='mixer']")
            page.fill("[data-ly-title]", "Old Song (Remix)")
            page.click("[data-action='ly-lookup']")
            page.wait_for_selector(".ly-line")
            check(Api.lookups and Api.lookups[-1].get("title") == "Old Song (Remix)" and Api.lookups[-1].get("artist") == "Old Band",
                  f"{h}: the lookup sends the edited artist and title ({Api.lookups[-1:]})")
            page.click("[data-action='mixview'][data-view='stems']")
            # The split song: Lead and Backing lanes instead of Vocals.
            page.click("[data-action='tab'][data-tab='library']")
            page.click(".track-load[data-id='c1b2c3d4e5f6']")
            page.wait_for_selector(".t-play:not([disabled])", timeout=15000)
            lanes = page.eval_on_selector_all("[data-action='mute']", "els => els.map(e => e.dataset.id)")
            check("lead_vocals" in lanes and "backing_vocals" in lanes and "vocals" not in lanes, f"{h}: split lanes replace Vocals ({lanes})")
            # Extract, with the mini player showing: the button must be clear of it.
            page.click(".t-play")
            page.click("[data-action='tab'][data-tab='extract']")
            page.wait_for_selector(".mini")
            check("beat model" in page.inner_text(".model-note"), f"{h}: first-download note on Extract")
            check(any(x.get("auto_sections") is False for x in Api.puts), f"{h}: a sections choice left on is switched off at start")
            check(page.get_attribute("[data-action='auto-sections']", "aria-pressed") == "false", f"{h}: sections off by default")
            page.click("[data-action='auto-sections']")
            page.wait_for_timeout(200)
            check(page.get_attribute("[data-action='auto-sections']", "aria-pressed") == "true", f"{h}: sections toggle on")
            check(Api.settings.get("auto_sections") is True, f"{h}: sections choice saved to the app")
            check("section model" in page.inner_text(".model-note"), f"{h}: section model download mentioned")
            page.click("[data-action='vocalmode'][data-mode='split']")
            check("on" in (page.get_attribute("[data-action='vocalmode'][data-mode='split']", "class") or ""), f"{h}: Lead + Backing selectable")
            check("vocal split model" in page.inner_text(".model-note"), f"{h}: vocal split model download mentioned")
            page.click("[data-action='vocalmode'][data-mode='all']")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(200)
            cta = page.locator("[data-action='split']").bounding_box()
            mini = page.locator(".mini").bounding_box()
            check(cta["y"] + cta["height"] <= mini["y"] + 0.5, f"{h}: Extract button above the mini player ({cta['y'] + cta['height']:.0f} <= {mini['y']:.0f})")
            if shots:
                page.screenshot(path=str(shots / f"extract-{h}.png"), full_page=False)
            check(not errors, f"{h}: no script errors {errors}")
            check(not outside, f"{h}: no requests leave the phone {outside[:3]}")
            page.close()
        browser.close()
    server.shutdown()
    print("\nAll UI checks passed." if not failures else f"\n{len(failures)} UI check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
