#!/usr/bin/env python3
"""Drives StemDeck in an Android emulator the way a person would, and records
what happens: screenshots, UI trees, whether the app process is alive, and the
crash buffer. Everything lands in OUT (default: emulator-out/).

Scenarios, in order:
  1. cold start
  2. Extract → Upload audio file → pick the test MP3 in the system picker →
     Extract stems, then watch the separation for a few minutes
  3. Share → StemDeck with the same file
  4. force-stop and start again (the "won't start any more" report)
"""
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PKG = "org.stemdeckport.android"
ACT = f"{PKG}/.ui.MainActivity"
OUT = os.environ.get("OUT", "emulator-out")
SONG = os.environ.get("SONG", "StemDeck Test Song.mp3")
WATCH_SECONDS = int(os.environ.get("WATCH_SECONDS", "300"))
os.makedirs(OUT, exist_ok=True)
step_no = 0
events = []


def adb(*args, check=False, timeout=60):
    r = subprocess.run(["adb", *args], capture_output=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"adb {' '.join(args)}: {r.stderr.decode(errors='replace')}")
    return r.stdout.decode(errors="replace")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    events.append(line)


def pid():
    return adb("shell", "pidof", PKG).strip()


def snap(name):
    """Screenshot + UI tree + liveness, numbered so they sort in order."""
    global step_no
    step_no += 1
    base = os.path.join(OUT, f"{step_no:02d}-{name}")
    with open(base + ".png", "wb") as f:
        f.write(subprocess.run(["adb", "exec-out", "screencap", "-p"], capture_output=True).stdout)
    xml = ui_xml()
    with open(base + ".xml", "w") as f:
        f.write(xml)
    focus = re.search(r"mCurrentFocus=\S+ \S+ (\S+)\}", adb("shell", "dumpsys", "window"))
    log(f"snapshot {step_no:02d}-{name}: pid={pid() or 'NOT RUNNING'} focus={focus.group(1) if focus else '?'}")
    return xml


def ui_xml():
    for _ in range(3):
        adb("shell", "uiautomator", "dump", "/sdcard/ui.xml")
        xml = adb("shell", "cat", "/sdcard/ui.xml")
        if xml.startswith("<?xml"):
            return xml
        time.sleep(1)
    return ""


def nodes(xml):
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    return list(root.iter("node"))


def find(xml, *needles):
    """First node whose text or content-desc contains one of needles."""
    for n in nodes(xml):
        label = (n.get("text") or "") + " " + (n.get("content-desc") or "")
        if any(k.lower() in label.lower() for k in needles):
            m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", n.get("bounds", ""))
            if m:
                x1, y1, x2, y2 = map(int, m.groups())
                return (x1 + x2) // 2, (y1 + y2) // 2, label.strip()
    return None


def tap_text(*needles, wait=15):
    end = time.time() + wait
    while time.time() < end:
        hit = find(ui_xml(), *needles)
        if hit:
            x, y, label = hit
            adb("shell", "input", "tap", str(x), str(y))
            log(f"tapped '{label}' at {x},{y}")
            time.sleep(1.5)
            return True
        time.sleep(1)
    log(f"could not find any of {needles} on screen")
    return False


def dismiss_dialogs():
    for label in ("Allow", "Dismiss", "OK", "Got it"):
        hit = find(ui_xml(), label)
        if hit:
            adb("shell", "input", "tap", str(hit[0]), str(hit[1]))
            log(f"dismissed a dialog with '{hit[2]}'")
            time.sleep(1)


def watch(seconds, name):
    """Snapshot every 30 s until the time is up or the app dies."""
    end = time.time() + seconds
    i = 0
    while time.time() < end:
        time.sleep(30)
        i += 1
        snap(f"{name}-{i * 30}s")
        if not pid():
            log("the app process is gone")
            return False
    return True


def media_uri(display_name):
    out = adb("shell", "content", "query", "--uri", "content://media/external/audio/media",
              "--projection", "_id:_display_name")
    for line in out.splitlines():
        if display_name in line:
            m = re.search(r"_id=(\d+)", line)
            if m:
                return f"content://media/external/audio/media/{m.group(1)}"
    return None


def main():
    adb("logcat", "-c")
    adb("logcat", "-b", "crash", "-c")

    log("=== 1. cold start")
    adb("shell", "am", "start", "-W", "-n", ACT)
    time.sleep(6)
    dismiss_dialogs()
    time.sleep(4)
    snap("cold-start")

    log("=== 2. import through Extract → Upload audio file")
    tap_text("Extract") or None
    snap("extract-tab")
    if tap_text("Upload audio file", "Upload audio"):
        time.sleep(4)
        snap("picker-open")
        # DocumentsUI: open the drawer's Downloads/Audio if the file is not listed yet.
        if not find(ui_xml(), SONG.rsplit(".", 1)[0]):
            for root in ("Downloads", "Download", "Audio"):
                if tap_text(root, wait=3):
                    time.sleep(2)
                    if find(ui_xml(), SONG.rsplit(".", 1)[0]):
                        break
            snap("picker-browsed")
        if tap_text(SONG.rsplit(".", 1)[0], wait=10):
            time.sleep(3)
            snap("file-chosen")
            if tap_text("Extract stems", wait=10):
                time.sleep(5)
                snap("extract-started")
                watch(WATCH_SECONDS, "separating")
    snap("after-upload-flow")

    log("=== 3. import through Share → StemDeck")
    uri = media_uri(SONG)
    log(f"media uri: {uri}")
    if uri:
        adb("shell", "am", "start", "-W", "-a", "android.intent.action.SEND", "-t", "audio/mpeg",
            "--eu", "android.intent.extra.STREAM", uri, "--grant-read-uri-permission", "-n", ACT)
        time.sleep(10)
        snap("after-share")
        watch(90, "share-separating")

    log("=== 4. force-stop and start again")
    adb("shell", "am", "force-stop", PKG)
    time.sleep(2)
    adb("shell", "am", "start", "-W", "-n", ACT)
    time.sleep(10)
    snap("restart")
    time.sleep(20)
    snap("restart-30s")

    with open(os.path.join(OUT, "logcat-crash.txt"), "w") as f:
        f.write(adb("logcat", "-d", "-b", "crash"))
    with open(os.path.join(OUT, "logcat.txt"), "w") as f:
        f.write(adb("logcat", "-d", "-v", "threadtime", timeout=120))
    with open(os.path.join(OUT, "exit-info.txt"), "w") as f:
        f.write(adb("shell", "dumpsys", "activity", "exit-info", PKG, timeout=60))
    with open(os.path.join(OUT, "events.txt"), "w") as f:
        f.write("\n".join(events) + "\n")
    log("done")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(OUT, "events.txt"), "w") as f:
            f.write("\n".join(events) + "\n")
    sys.exit(0)
