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
    }


JOBS = [job("a1b2c3d4e5f6", "Grid Song", True), job("b1b2c3d4e5f6", "Old Song", False)]


class Api(BaseHTTPRequestHandler):
    settings = {"stem_format": "mp3", "gpu": True, "click": {}}
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
        if self.path == "/api/settings":
            Api.puts.append(body)
            Api.settings.update({k: v for k, v in body.items() if k != "click"})
            if "click" in body:
                Api.settings["click"] = body["click"]
            return self.send(200, json.dumps(Api.settings).encode(), "application/json")
        self.send(404, b"{}", "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/mobile":
            return self.send(302, b"", "text/plain", {"Location": "/mobile/"})
        if path == "/api/config":
            return self.send(200, json.dumps({"platform": "android", "models": {"separation": True, "beat": False}}).encode(), "application/json")
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
            # A song with no grid: no click controls, and the reason shown.
            page.click("[data-action='mixview'][data-view='stems']")
            page.click("[data-action='tab'][data-tab='library']")
            page.click(".track-load[data-id='b1b2c3d4e5f6']")
            page.wait_for_selector(".t-play:not([disabled])", timeout=15000)
            check(page.locator(".click-row").count() == 0, f"{h}: no click controls without a grid")
            check("only 3 beats" in page.inner_text(".click-note"), f"{h}: says why there is no grid")
            # Extract, with the mini player showing: the button must be clear of it.
            page.click(".t-play")
            page.click("[data-action='tab'][data-tab='extract']")
            page.wait_for_selector(".mini")
            check("beat model" in page.inner_text(".model-note"), f"{h}: first-download note on Extract")
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
