#!/usr/bin/env bash
# Creates and boots an Android emulator on a GitHub runner, logging everything
# to emulator-out/ so a failed boot can be diagnosed.
set -u
OUT=${OUT:-emulator-out}
mkdir -p "$OUT"
API=${API:-34}
IMG="system-images;android-$API;google_apis;x86_64"
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-/usr/local/lib/android/sdk}}
export PATH="$SDK/cmdline-tools/latest/bin:$SDK/emulator:$SDK/platform-tools:$PATH"
# One explicit place for the AVD, for both avdmanager and the emulator.
export ANDROID_USER_HOME="$HOME/.android"
export ANDROID_AVD_HOME="$HOME/.android/avd"
mkdir -p "$ANDROID_AVD_HOME"

{
  echo "== host"; nproc; free -g; ls -l /dev/kvm
  echo "== sdk $SDK"
} > "$OUT/boot-host.txt" 2>&1

yes | sdkmanager --licenses > /dev/null 2>&1
sdkmanager --install "platform-tools" "emulator" "$IMG" > "$OUT/sdkmanager.txt" 2>&1
emulator -accel-check >> "$OUT/boot-host.txt" 2>&1

echo no | avdmanager create avd -n test -k "$IMG" -d pixel_6 -p "$ANDROID_AVD_HOME/test.avd" --force >> "$OUT/boot-host.txt" 2>&1
CFG="$ANDROID_AVD_HOME/test.avd/config.ini"
if [ ! -f "$CFG" ]; then
  echo "AVD was not created" | tee -a "$OUT/boot-host.txt"
  ls -la "$ANDROID_AVD_HOME" "$HOME/.android" >> "$OUT/boot-host.txt" 2>&1
  exit 1
fi
# The runner has 2 cores and 7 GB: keep the guest small.
{
  echo "hw.ramSize=3072"
  echo "disk.dataPartition.size=8G"
  echo "hw.cpu.ncore=2"
  echo "hw.keyboard=yes"
} >> "$CFG"
emulator -list-avds >> "$OUT/boot-host.txt" 2>&1

nohup emulator -avd test -no-window -no-audio -no-boot-anim -no-snapshot \
  -gpu swiftshader_indirect -accel on -camera-back none -camera-front none \
  > "$OUT/emulator.log" 2>&1 &

adb start-server
start=$(date +%s)
booted=0
while [ $(( $(date +%s) - start )) -lt 1200 ]; do
  if [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; then
    booted=1; break
  fi
  sleep 10
done
echo "boot_completed=$booted after $(( $(date +%s) - start )) s" | tee -a "$OUT/boot-host.txt"
adb devices >> "$OUT/boot-host.txt" 2>&1
[ "$booted" = 1 ] || exit 1
sleep 20  # let the launcher and system services settle
adb shell input keyevent 82
