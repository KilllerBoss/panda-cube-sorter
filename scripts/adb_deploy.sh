#!/usr/bin/env bash
# adb_deploy.sh — APK installieren, App starten, Logs live mitschneiden
set -e
cd "$(dirname "$0")/.."
APK=apk/PandaCubeSorter-v1.0.0-release.apk
[ -f "$APK" ] || APK=app/build/outputs/apk/release/app-release.apk

echo "== install $APK =="
adb install -r "$APK"

echo "== start app =="
adb shell am start -n dev.pandasorter/android.app.NativeActivity

echo "== live logs (Ctrl-C zum Beenden) =="
adb logcat -s panda-sorter:V
