# Architektur — von den Rohdaten zum Drehmoment

Dieses Dokument beschreibt den Datenfluss durch das System pro 100-Hz-Zyklus
(10 ms Budget), die Zahlenformate der Gewichte und die Design-Entscheidungen
mitsamt ihren Begründungen.

---

## 1. Gesamtübersicht (ein Zyklus, 10 ms)

```
[0.0 ms]  mj_step × 5 (dt = 2 ms)            Physik-Substeps, Budget ~0,8 ms
[0.8 ms]  Event-Kamera (96×72 GLES-FBO)      |ΔL| > 0,045 → bipolare Events,
          → 12×8-Zelle × (2 Pol + 4 Hue)      576 Bins
[1.0 ms]  ALIF-LSNN (128 Neuronen)           v' = λ·v + W·bins + b
          adaptiver Schwellwert               θ_i = vth0·(1+β·a_i)
[1.5 ms]  Predictive Coding (FEP)            û = W_fb·a ; e = u − û
          → 32-dim Embedding                  emb = ReLU(W_pr·[e ; a])
[1.7 ms]  Detektionskopf (Pulse-Frame)       8 Slots: (x, y, Farbe, Score)
[2.0 ms]  Task-Layer (State-Machine)         HOME→HOVER→DESCEND→GRASP→LIFT→
          → Phase, s, tcp-Ziel                TRANSPORT→PLACE (ereignisgetrieben)
[2.2 ms]  Soft-MoE-Gating                    bias = W_sm·emb + b_sm (±1,5)
[2.3 ms]  KAN-MLP (kompiliert)               x = [Δq, q_start, s, grip, Phase]
          + LoRA-Addition                     logits = W2·h + b + bias + B·A·h1
[2.5 ms]  Prototypen-Blending                q_add = Σ_r w_r·P_r(s)
[2.7 ms]  Lyapunov-Update                    A,B ← Regel aus Tracking-Fehler e
[3.0 ms]  Torques → mjData->ctrl             τ = qfrc_bias + kp(q*−q) + kd(q̇*−q̇)
[3.1 ms]  Puffer / Sleep bis 10 ms           ~7 ms Reserve (Thermik!)
```

Alle Komponenten laufen in **einem nativen C++-Thread**; der Embedding-Vektor
wird ohne Kopie als Zeiger an die nachgelagerten Schichten übergeben
(Zero-Copy-Pipeline: `ControllerInput`/`ControllerOutput` sind flache
Structs mit festen Arraygrößen).

## 2. Physik-Fundament (Phase 1)

- **MuJoCo 3.13.0** wird vom NDK r27b für arm64-v8a gebaut
  (`scripts/build_mujoco_android.sh`):
  `-O3 -fPIC -ffp-contract=fast` — bewusst **ohne** `-ffast-math`, da
  `-ffast-math` die Konvergenz des Newton-Solvers und die Reproduzierbarkeit
  der Kontaktauflösung gefährdet. Zwei NDK-spezifische Anpassungen sind im
  Skript dokumentiert: `_POSIX_C_SOURCE=200809L` (thread-safe `localtime`)
  und ein `ld.gold`-Shim (r27 liefert nur lld).
- **Szene** (`scene/scene_template.xml`): Panda-artiger 7-DOF-Arm mit
  Franka-Gelenkkonvention (j1 z, j2/j4/j6 y, j3/j5/j7 x) und Franka-Längen
  (0,333 / 0,316 / 0,0825 / 0,384 / 0,088 / 0,107 m). Bewusste Vereinfachungen
  gegenüber dem echten Panda:
  - Geometrie aus Kapseln/Boxen statt Meshes (schnelleres mj_step, kleine MJB);
  - j4-Bereich symmetrisch (±3,07 statt −3,07…−0,07), damit die
    Ellbogen-oben-Faltung der Closed-Form-IK verfügbar ist;
  - Arm-Arm-Selbstkollision via contype/conaffinity ausgeschaltet
    (Arm↔Würfel, Arm↔Tisch, Würfel↔Tisch bleiben aktiv).
- **Tischhöhe 0,25 m** und Wurf-/Zonen-Annulus sind so gewählt, dass die
  Closed-Form-IK (unten) für alle Greifziele gültige, kollisionsfreie
  Konfigurationen liefert.
- **Binäres MJB:** `scene/compile_scene.py` injiziert N Würfel (8 aktiv /
  50 Stresstest) und speichert `scene_8.mjb`/`scene_50.mjb`. MJB ist
  plattformunabhängig (little-endian) — auf dem Gerät gibt es kein
  XML-Parsing. Die App setzt `mjData->qpos` pro Episode direkt (Zufalls-
  würfel mit Mindestabstand und Zonen-Ausschluss, Zufalls-Gierwinkel).

## 3. Visuelle Wahrnehmung (Phase 2)

### Event-Kamera (`event_camera.cpp`)

- Eingabe: RGB-Frame 96×72 aus dem GLES-FBO (App) bzw. CPU-Projektor (Desktop).
- Log-Helligkeitsdifferenz `ΔL = L(t) − L(t−1)`; nur Pixel mit `|ΔL| > 0,045`
  erzeugen bipolare Events (+1/−1). Ruhe erzeugt null Daten.
- Zusätzlich 4 Hue-Buckets (r/g/b/gelb) pro geändertem Pixel → zusammen
  12×8-Zellen × 6 Kanäle = **576 Bins** (Eingabe des LSNN).
- **Refresh-Puls:** Statische Szenen erzeugen keine Events (DVS-Physik).
  Alle 500 ms verschiebt die Kamera ihr Bild um ~8 mm (Puls) — die gesamte
  Szene emittiert erneut Events und der Detektionskopf erhält ein frisches
  „Standbild". Der Puls ist im Traces als `refresh_pulse` sichtbar.

### ALIF-LSNN (`alif_lsnn.cpp`)

- 128 Neuronen, dichte Eingangsprojektion (576×128, fp32, fest im Blob).
- Adaptiver Schwellwert: `θ_i = vth0·(1+β·a_i)` mit Spur `a ← ρ·a + spike`
  — das ist das Kurzzeitgedächtnis für Bewegungsverläufe (hohe a_i ⇒
  weniger Output ⇒ Adaptation an Reize, die schon „erklärt" sind).
- Reset bei Spike um θ (Subtraktion), λ = 0,85 Leak.

### Predictive Coding (FEP, `pred_coder.cpp`)

- Der Zustand `a` **vorhersagt** das nächste Eingangsmuster:
  `û = W_fb·a`. Nur der Fehler `e = u − û` wird weitergeleitet
  („nur Überraschung propagiert").
- Lokale, begrenzte Hebb-Regel online: `W_fb += η·a·eᵀ / (1+a²)`,
  Werte auf ±0,5 geklemmt — eine Closed-Form-artige lokale Regel ohne
  Globalgradienten.
- Embedding: `emb = ReLU(W_proj·[e ; a])`, festes (seeded) 704×32-Projektion.

## 4. Motorik-Synthese (Phase 3)

### Bewegungsbibliothek & NMF

`toolchain/step1_collect.py` fährt einen skriptierten Experten (echte
Würfelpositionen, ereignisgetriebene Phasen, Rate-Limiter 1,5 rad/s,
Closed-Form-IK) und zeichnet pro Zyklus Phase, Fortschritt s, q_start,
q_goal, q_des, Embedding und Pulse-Frame auf. `step2_nmf.py` bildet die
Residuen `q_des − [q_start + (q_goal−q_start)·minjerk(s)]` pro Bewegungs-
segment auf 64 Samples ab und faktorisiert sie mit Multiplicative-Update-NMF
(Rang 8) → **Prototypen-Bank P[8][7][64]** (nur nichtnegative Hülle).

### KAN → exaktes ReLU-MLP

- Layer 1 (24→8) und Layer 2 (8→8) sind **additive univariate
  Splines-Modelle**: jede Kante ist eine stückweise lineare Funktion mit 5
  Knoten (Quantile der Trainingsdaten), gefittet per Ridge auf der
  Hinge-Basis `[x ; ReLU(x−t_k)]`.
- Die Runtime-Identität `f(x) = c₀ + c₁x + Σ cₖ·ReLU(x−tₖ)` macht aus dem
  KAN **ohne Approximationsfehler** ein dichtes MLP:
  `h = W1ᵀ·[x ; ReLU(x−t1)] + b1`, `logits = W2ᵀ·[h ; ReLU(h−t2)] + b2`.
  `step4_kan_to_mlp.py` verifiziert die Kompilierung numerisch
  (max. Differenz 0,0 in der letzten Ausführung).
- Der Kopf gibt **Mischgewichte w (8, Softmax)** über die Prototypen aus.
- Die Zielgröße w* wird pro Zyklus aus der Projektion des Experten-Residuum
  auf die bei s gesampelte Prototypen-Bank bestimmt (nichtnegativ, normiert).

### Soft-MoE

`bias = W_sm·emb + b_sm` (mit tanh auf ±1,5 begrenzt) wird auf die KAN-Logits
addiert: das Seh-Embedding moduliert kontinuierlich die Mischung zwischen
Annäherungs-, Greif- und Ablage-Prototypen. Training: Ridge auf den
Residual-Logits `log(w*) − logits_KAN` (step5).

### Ablage-Geometrie

`q*(s) = q_start + (q_goal − q_start)·minjerk(s) + Σ_r w_r·P_r[s]`
— die geometrische Basis bleibt immer erhalten; Prototypen und LoRA liefern
nur Texture (±0,15 rad Clamps), d. h. ein falsches Gate kann die Bahn
verziehen, aber nie verlieren.

## 5. Inferenz & Hardware-Entscheidung (Phase 4)

- Der kompilierte Kopf braucht pro Zyklus nur zwei GEMV-Aufrufe
  (120×8 und 40×8) — rein fp32, NEON-freundlich, keine Splines, keine
  dynamische Allokation.
- `--mode bench` des Harness misst Warmup (1.000 Zyklen) und dann 2.000
  Zyklen: `mj_step`-Budget und Gesamtzyklus. Die Entscheidungsregel
  „CPU wenn inkl. Speichertransfer schneller als NPU plus Dispatch"
  ist als Vergleich CPU-Pfad vs. gemessenem Dispatch-Overhead implementiert;
  die QNN-Adapter (Hexagon, INT8) sind als Hook vorbereitet
  (`docs/BENCHMARKS.md`, Abschnitt NPU).
- FP16-Pfad: Der MLP-Forward ist strukturell identisch in `__fp16`
  ausführbar; auf dem S26 Ultra lohnt der Wechsel erst, wenn der
  Transfer-Domänenwechsel (CPU→NPU→CPU) die Rechenzeit dominiert — genau
  das misst das Benchmark-Skript.

## 6. Echtzeit-Adaption (Phase 5)

`logits = W2·h1 + b2 + B·A·h1` mit `A ∈ R^{8×4}`, `B ∈ R^{4×8}` (Rang 4,
fp32, CPU).

Update (kein Backprop), pro Zyklus aus dem Gelenk-Fehler `e = q* − q`:

```
ΔB += η·e·(Aᵀh1)ᵀ          ΔA += η·(B e)·h1ᵀ          (normierte Gradienten)
η = η0 / (1 + ‖h1‖)        (Skalierung)
V = eᵀe ;  wenn V steigt → η ← 0,7·η  sonst η ← min(1,0, 1,02·η)   (Lyapunov-Gate)
‖A‖F ≤ 1,5 , ‖B‖F ≤ 1,5    (harte BIBO-Schranken)
```

Damit ist die Adaption divergence-frei: Sinkt der Tracking-Fehler nicht,
wird die Schrittweite gedrosselt, bevor die Instabilität das System kippt.
Die Basis-MLP-Gewichte bleiben eingefroren; nur A, B und der FEP-Feedback
`W_fb` sind plastisch.

## 7. APK-Integration (Phase 6)

- **NativeActivity** (reines C++, `android:hasCode="false"`) — keine
  WebView, kein JNI-Overhead, keine Java-Klasse.
- **EGL/GLES 3.0**: Ein Kontext, zwei Renderpfade auf einem Thread-Paar:
  - *Render-Thread*: interaktive Ansicht (`cam_track`) + HUD bei Display-Rate.
  - *Loop-Thread*: Event-Kamera-FBO (96×72) + `mj_step`×5 + Regelung, pacing
    auf 100 Hz (`sleep_until`), Überläufe werden geloggt (Thermik-Wächter).
- **Assets**: `scene.mjb` (67 KB), `scene_50.mjb` (177 KB), `weights.bin`
  (820 KB). Die MJB wird an `mj_loadModel` über einen temporären Pfad
  übergeben (mj_loadModel liest nur Dateien).
- **Minimales UI**: Tippen = neue Episode (neues zufälliges Würfellayout);
  Statusleiste unten = genutzter Zyklusbudget-Anteil; adb logcat liefert
  die Detail-Telemetrie.

## 8. Zahlenformate

- **weights.bin**: `PCSW`-Magic, Version 1, little-endian, Sektionstabelle
  (`PROTO`, `LSNN_W/B/G`, `PRED_FB`, `EMB_PROJ`, `SMOE_W/B`, `KAN_W1/B1/W2/B2`,
  `HINGE_T1/T2`, `DEC_W/B`, `LORA_A/B`, `IN_SCALE`, `META`). Der Loader
  prüft jede Sektion gegen die Compile-Zeit-Konstanten in `types.h`.
- **META** (8 floats): `λ_v, ρ_a, β_θ, vth0, snn_η, pred_η, out_scale, resv`.
- Alle Matrizen row-major fp32; der Decoder-Kopf liest
  `[embedding(32) ; gelatchte Pulse-Bins(576)]` (Sensorfusion am Kopf —
  die Pulse liefern das Standbild, das Embedding die Dynamik).
