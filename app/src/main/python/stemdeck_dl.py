"""Link import for StemDeck for Android (1.4): desktop StemDeck's download.py
(app/pipeline/download.py) as far as the phone needs it.

yt-dlp is limited to the YouTube and SoundCloud extractors (the "generic"
one stays out: desktop's SSRF boundary, #173), fetches the best audio
stream as it is (no FFmpeg on the phone; Android decodes m4a/webm/opus/mp3),
probes the length first and refuses songs over the cap, retries transient
network errors with back-off, and solves YouTube's JavaScript challenges
with QuickJS-NG (libqjs.so, shipped as a native library) and yt-dlp-ejs.

The Kotlin side has already validated and normalised the URL (LinkUrl.kt,
desktop's validate_youtube_url) before it gets here.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

# An updated yt-dlp downloaded from the app's Settings (a zipapp, which is
# importable as is) wins over the one in the APK.
_UPDATE = os.environ.get("STEMDECK_YTDLP_UPDATE", "")
if _UPDATE and os.path.isfile(_UPDATE) and _UPDATE not in sys.path:
    sys.path.insert(0, _UPDATE)

from yt_dlp import YoutubeDL  # noqa: E402
from yt_dlp.version import __version__ as YTDLP_VERSION  # noqa: E402

ALLOWED_EXTRACTORS = ["youtube", "soundcloud"]
SOCKET_TIMEOUT_SEC = 30
MAX_RETRIES = 3
RETRY_BACKOFF = (2, 4, 8)
_RETRIABLE = ("connection reset", "ssl", "timed out", "network is unreachable", "temporary failure",
              "unable to download", "read timed out", "remotedisconnected", "broken pipe", "connection refused")
_NON_RETRIABLE = ("private video", "video unavailable", "has been removed", "http error 404", "http error 403",
                  "not available in your country", "age-restricted")


class Cancelled(Exception):
    pass


def _retriable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(s in msg for s in _NON_RETRIABLE):
        return False
    return any(s in msg for s in _RETRIABLE)


def _opts(qjs: str) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "allowed_extractors": ALLOWED_EXTRACTORS,
        "socket_timeout": SOCKET_TIMEOUT_SEC,
        "noplaylist": True,
        "cachedir": False,
    }
    if qjs and os.path.isfile(qjs):
        opts["js_runtimes"] = {"quickjs": {"path": qjs}}
    return opts


def _with_retries(fn, cancelled):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 - yt-dlp raises DownloadError for everything
            if cancelled() or isinstance(getattr(e, "exc_info", (None, None))[1], Cancelled):
                raise Cancelled() from e
            if attempt == MAX_RETRIES or not _retriable(e):
                raise
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            for _ in range(int(wait * 10)):
                if cancelled():
                    raise Cancelled() from e
                time.sleep(0.1)


def runtime_info(qjs: str) -> str:
    """What the app reports in Settings: yt-dlp's version and the JS runtime it found."""
    from yt_dlp.utils._jsruntime import QuickJsRuntime

    info = QuickJsRuntime(qjs).info if qjs and os.path.isfile(qjs) else None
    return json.dumps({
        "yt_dlp": YTDLP_VERSION,
        "js_runtime": f"{info.name} {info.version}" if info else None,
        "updated": bool(_UPDATE and _UPDATE in sys.path),
    })


def download(url: str, out_dir: str, qjs: str, max_duration: float, listener) -> str:
    """Downloads the audio of ``url`` into ``out_dir`` as input.<ext>. Returns
    JSON {file, title, duration, thumbnail, artist, track, album}. ``listener``
    is the Kotlin side: onProgress(fraction, text) -> keep going."""

    def cancelled() -> bool:
        return not listener.onProgress(-1.0, "")

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                p = float(d.get("downloaded_bytes", 0)) / float(total)
                if not listener.onProgress(p, f"Downloading {int(p * 100)}%"):
                    raise Cancelled()
            elif cancelled():
                raise Cancelled()
        elif d.get("status") == "finished":
            listener.onProgress(1.0, "Download complete")

    listener.onProgress(0.0, "Looking up the link")

    def probe() -> dict:
        with YoutubeDL(_opts(qjs)) as ydl:
            return ydl.extract_info(url, download=False) or {}

    meta = _with_retries(probe, cancelled)
    duration = meta.get("duration") or 0
    if max_duration and duration > max_duration:
        raise RuntimeError(f"The song is {int(duration // 60)} min long; the limit is {int(max_duration // 60)} min.")

    for old in glob.glob(os.path.join(out_dir, "input.*")):
        os.remove(old)

    def fetch() -> dict:
        opts = {
            **_opts(qjs),
            # The best audio stream as it is: m4a first, which every Android decodes.
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(out_dir, "input.%(ext)s"),
            "noprogress": True,
            "progress_hooks": [hook],
        }
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=True) or {}

    info = _with_retries(fetch, cancelled)
    files = sorted(f for f in glob.glob(os.path.join(out_dir, "input.*")) if not f.endswith(".part"))
    if not files:
        raise RuntimeError("yt-dlp finished but wrote no audio file")
    artists = info.get("artists")
    return json.dumps({
        "file": files[0],
        "title": info.get("title") or meta.get("title"),
        "duration": info.get("duration") or duration or None,
        "thumbnail": info.get("thumbnail") or meta.get("thumbnail"),
        "artist": info.get("artist") or (artists[0] if isinstance(artists, list) and artists else None),
        "track": info.get("track"),
        "album": info.get("album"),
        "extractor": info.get("extractor_key"),
    })
