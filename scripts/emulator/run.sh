#!/usr/bin/env bash
# Runs inside the emulator job: installs the APK, puts a test MP3 on the
# device and drives the app (drive.py). Never fails the step itself, so the
# logs are always published; the summary says what happened.
set -u
OUT=${OUT:-emulator-out}
mkdir -p "$OUT"
SONG="StemDeck Test Song.mp3"

adb wait-for-device
adb shell getprop ro.build.version.release | sed 's/^/Android /'
adb shell settings put global window_animation_scale 0
adb shell settings put global transition_animation_scale 0
adb shell settings put global animator_duration_scale 0

adb install -r -g StemDeck-Android.apk 2>&1 | tail -2

# A 3-minute test song as a 320 kbps MP3, like the files it will really get.
python3 scripts/emulator/make_song.py song.wav
ffmpeg -loglevel error -y -i song.wav -codec:a libmp3lame -b:a 320k "$SONG"
adb push "$SONG" "/sdcard/Download/$SONG" >/dev/null
adb shell content call --uri content://media --method scan_volume --arg external_primary >/dev/null 2>&1 || true
adb shell am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d "file:///sdcard/Download/${SONG// /%20}" >/dev/null 2>&1 || true
sleep 5

OUT="$OUT" SONG="$SONG" timeout 1500 python3 scripts/emulator/drive.py 2>&1 | tee "$OUT/drive.log"
exit 0
