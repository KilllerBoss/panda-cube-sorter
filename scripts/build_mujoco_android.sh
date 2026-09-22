#!/usr/bin/env bash
# build_mujoco_android.sh — cross-kompiliert MuJoCo 3.13 für arm64-v8a.
# Patches (für NDK r27 + lld-only):
#   1. _POSIX_C_SOURCE=200809L   (thread-safe localtime im Bionic)
#   2. ld.gold-Shim              (r27 entfernt gold; qhull-CMake probe)
#   3. simulate/tests/examples OFF
# Ergebnis: app/libs/arm64-v8a/libmujoco.so + Header unter third_party/.
set -e
BASE=/home/z/my-project
SDK=${ANDROID_HOME:-$BASE/android-sdk}
REPO=$BASE/download/panda-cube-sorter
NDK=$SDK/ndk/27.1.12297006
CMAKE_BIN=$SDK/cmake/3.22.1/bin/cmake
SRC=$BASE/tools/mujoco-src
BUILD=$BASE/tools/mujoco-build-arm64

[ -d "$SRC" ] || git clone --depth 1 --branch 3.13.0 \
  https://github.com/google-deepmind/mujoco.git "$SRC"

# ld.gold-Shim (ruft ld.lld) — für CMake-Probes, die gold erwarten
mkdir -p "$BASE/tools/bin"
for l in ld.gold gold ld.bfd; do
  printf '#!/bin/sh\nexec %s/bin/ld.lld "$@"\n' "$NDK/toolchains/llvm/prebuilt/linux-x86_64" \
    > "$BASE/tools/bin/$l"
  chmod +x "$BASE/tools/bin/$l"
done
export PATH="$BASE/tools/bin:$PATH"

FLAGS="-O3 -fPIC -ffp-contract=fast -D_POSIX_C_SOURCE=200809L"
rm -rf "$BUILD"
$CMAKE_BIN -S "$SRC" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF \
  -DBUILD_SHARED_LIBS=ON
$CMAKE_BIN --build "$BUILD" --target mujoco -j2

mkdir -p "$REPO/app/libs/arm64-v8a" "$REPO/app/src/main/cpp/third_party/mujoco/include"
cp "$BUILD/lib/libmujoco.so" "$REPO/app/libs/arm64-v8a/"
cp -r "$SRC/include/mujoco" "$REPO/app/src/main/cpp/third_party/mujoco/include/"
echo "OK: $REPO/app/libs/arm64-v8a/libmujoco.so"
