# Build-Guide — von Null zur installierten APK

Dieses Dokument führt durch jeden Schritt: SDK/NDK einrichten, MuJoCo
cross-kompilieren, Szene und Gewichte erzeugen, APK bauen, auf dem S26 Ultra
installieren. Alle Pfade beziehen sich auf das Repo-Wurzelverzeichnis.

---

## 0. Voraussetzungen

| Werkzeug | Version | Hinweis |
|---|---|---|
| JDK | 17+ | nur für Gradle/sdkmanager |
| Android SDK Platform | 34 | `platforms;android-34` |
| Build-Tools | 34.0.0 | |
| NDK | **r27b** (27.1.12297006) | r27 liefert nur lld — siehe 2.3 |
| CMake | 3.22.1 (SDK) | AGP-kompatibel |
| Python | 3.10+ | mit numpy + mujoco |

Alles automatisch: `bash scripts/setup_android_sdk.sh`
(setzt `ANDROID_HOME=/…/android-sdk`, lädt NDK + cmdline-tools + Pakete,
~1,5 GB — Resume-fähig via `curl -C -`).

## 1. Szene kompilieren (XML → MJB)

```bash
cd scene
python3 compile_scene.py        # erzeugt scene_8.mjb + scene_50.mjb (+ .xml)
python3 check_kinematics.py     # Reachability-Report der Closed-Form-IK
```

- `scene_8.mjb` (8 Würfel, 4 Farben × 2) = aktive Sortierszene.
- `scene_50.mjb` (50 Würfel) = Physik-/Thermik-Stresstest.
- `check_kinematics.py` muss mit `ALL_REACHABLE` enden — sonst sind Zonen/
  Spawn-Regionen außerhalb des gültigen Bereichs der Closed-Form-IK
  (siehe `docs/ARCHITEKTUR.md`, Abschnitt 8).

## 2. MuJoCo für arm64-v8a cross-kompilieren

### 2.1 Skript

```bash
bash scripts/build_mujoco_android.sh   # clang + NDK-Toolchain, -j2
# Ergebnis: app/libs/arm64-v8a/libmujoco.so + Header in app/src/main/cpp/third_party/
```

### 2.2 Compiler-Flags (fest im Skript)

```
-O3 -fPIC -ffp-contract=fast -D_POSIX_C_SOURCE=200809L
```

- `-ffp-contract=fast`: FMA-Fusion erlaubt (Schnelligkeit), **kein**
  `-ffast-math` (Newton-Solver-Konvergenz!).
- `-D_POSIX_C_SOURCE=200809L`: MuJoCo 3.13 verlangt thread-safe `localtime`
  (`localtime_r`), das Bionic nur mit diesem Makro freigibt.

### 2.3 NDK-r27-Besonderheiten (im Skript gepatcht)

1. `ld.gold` existiert in r27 nicht; qhulls CMake probe wird von einer
   **`ld.gold`-Shim** abgefangen (`tools/bin/ld.gold` → ruft `ld.lld`).
2. `MUJOCO_BUILD_SIMULATE=OFF`, `MUJOCO_BUILD_TESTS=OFF`,
   `MUJOCO_BUILD_EXAMPLES=OFF` — nur die Kernbibliothek wird gebaut.

### 2.4 Manuell (falls ohne Skript)

```bash
cmake -S tools/mujoco-src -B tools/mujoco-build-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/27.1.12297006/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-O3 -fPIC -ffp-contract=fast -D_POSIX_C_SOURCE=200809L" \
  -DCMAKE_CXX_FLAGS="-O3 -fPIC -ffp-contract=fast -D_POSIX_C_SOURCE=200809L" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build tools/mujoco-build-arm64 --target mujoco -j2
cp  tools/mujoco-build-arm64/lib/libmujoco.so app/libs/arm64-v8a/
cp -r tools/mujoco-src/include/mujoco app/src/main/cpp/third_party/mujoco/include/
```

## 3. Gewichte trainieren (volle Pipeline)

```bash
cd toolchain
pip install -r requirements.txt       # numpy, mujoco

python3 step0_init.py                 # feste Zufallsanteile (Seed 1234)
python3 step1_collect.py 4            # 4 Experten-Episoden (~30k Zyklen, ~5 min)
python3 step2_nmf.py                  # 8 Bewegungsprototypen (NMF)
python3 step3_kan_train.py            # KAN als additive Splines (Ridge)
python3 step4_kan_to_mlp.py           # exakte ReLU-Kompilierung + Verifikation
python3 step5_train_heads.py          # Detektor + Soft-MoE-Köpfe
python3 step6_export.py               # → weights/weights.bin (+ meta json)
```

Ergebnis in der letzten Ausführung:

- `step4`: `max|h_rt−h_ref| = 0.00e+00` (Kompilierung exakt)
- `step5`: Dekoder mittlerer Positionfehler **1,38 cm** (Pulse-Frame-Kopf)
- `weights.bin`: 819.692 Bytes, 20 Sektionen

Details: `docs/PIPELINE.md`.

## 4. APK bauen

```bash
bash scripts/build_apk.sh              # gradle assembleRelease --no-daemon
# Ergebnis: app/build/outputs/apk/release/app-release.apk
```

Der Release-Build ist mit dem beiliegenden `app/release.keystore` signiert
(Passwörter stehen in `app/build.gradle.kts` — für private Nutzung OK,
für Play-Store eigener Keystore!).

## 5. Installieren + Benchmarks

```bash
adb install -r apk/PandaCubeSorter-v1.0.0-release.apk
adb logcat -s panda-sorter            # Telemetrie (Zykluszeiten, Phasen)
bash scripts/adb_warmup_benchmark.sh  # Warmup + Timing auf dem Gerät
bash scripts/adb_deploy.sh            # Logs live mitschneiden
```

Messmethodik und Zielwerte: `docs/BENCHMARKS.md`.

## 6. Desktop-Harness (ohne Android testen)

```bash
cmake -S desktop -B desktop/build \
  -DMUJOCO_DIR=$HOME/.venv/lib/python3.12/site-packages/mujoco \
  -DCMAKE_BUILD_TYPE=Release
cmake --build desktop/build -j2
./scripts/run_harness.sh --mode bench    # Warmup + Timing (x86)
./scripts/run_harness.sh --mode eval --episodes 5
./scripts/run_harness.sh --mode stress --scene scene/scene_50.mjb
```

Der Harness nutzt denselben C++-Kern wie die App; nur die Event-Kamera
kommt aus einem CPU-Projektor statt dem GLES-FBO.

## 7. Fehlerbehebung (häufige Fallstricke)

| Symptom | Ursache | Lösung |
|---|---|---|
| `localtime`-`#error` beim MuJoCo-Build | fehlendes `_POSIX_C_SOURCE` | Flags aus Abschnitt 2.2 |
| `invalid linker name '-fuse-ld=gold'` | NDK r27 hat kein gold | Skript 2.1 (Shim) |
| `scene.mjb not found` | Assets nicht kopiert | Abschnitt 3 nochmal |
| `weights load failed: …` | Blob/Code-Konstanten gemischt | Pipeline komplett neu laufen lassen (0→6) |
| Episoden enden sofort | Detektor liefert keine Slots | Pulse-Frame prüfen (`logs/pulse_frame.ppm`, Dump-Env `PCS_DUMP`) |
| Arm friert in Home | q_des-Synthese nach ctrl-Write | Reihenfolge in `sim_glue.cpp` prüfen (Commit vor Torques) |
