#!/usr/bin/env bash
# setup_android_sdk.sh — lädt NDK r27b, cmdline-tools, platform 34,
# build-tools 34, cmake 3.22.1 und platform-tools. Resume-fähig.
set -uo pipefail
BASE=/home/z/my-project
SDK=$BASE/android-sdk
DL=$BASE/downloads
mkdir -p "$SDK/ndk" "$DL"

echo "[1/4] NDK r27b (Resume-fähig)..."
curl -sSL --retry 5 -C - -o "$DL/ndk.zip" \
  https://dl.google.com/android/repository/android-ndk-r27b-linux.zip
unzip -q "$DL/ndk.zip" -d "$SDK/ndk" && rm -f "$DL/ndk.zip"
mv "$SDK/ndk/android-ndk-r27b" "$SDK/ndk/27.1.12297006"

echo "[2/4] cmdline-tools..."
curl -sSL --retry 5 -C - -o "$DL/clt.zip" \
  https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip -q -o "$DL/clt.zip" -d "$DL/clt" && rm -f "$DL/clt.zip"
mkdir -p "$SDK/cmdline-tools" && mv "$DL/clt/cmdline-tools" "$SDK/cmdline-tools/latest"

export ANDROID_HOME=$SDK
export PATH=$SDK/cmdline-tools/latest/bin:$PATH
echo "[3/4] Lizenzen + Pakete..."
yes | sdkmanager --licenses >/dev/null
sdkmanager --install "platforms;android-34" "build-tools;34.0.0" "cmake;3.22.1" "platform-tools"

echo "[4/4] Gradle 8.9..."
curl -sSL --retry 5 -C - -o "$DL/gradle.zip" \
  https://services.gradle.org/distributions/gradle-8.9-bin.zip
mkdir -p "$BASE/tools" && unzip -q -o "$DL/gradle.zip" -d "$BASE/tools" && rm -f "$DL/gradle.zip"

echo "FERTIG. ANDROID_HOME=$SDK"
