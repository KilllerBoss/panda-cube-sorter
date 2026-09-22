# Trainings-Pipeline — wie die Gewichte entstehen

Die Pipeline ist deterministisch (Seed 1234/7/5) und ohne GPU lauffähig.
Sie erzeugt `weights/weights.bin`, das einzige Modellartefakt im APK.

```
step0_init.py        feste Zufallsanteile (LSNN-Projektion, FEP-Feedback-Init,
                     Embedding-Projektion, LoRA-Nullen, META-Skalare)
step1_collect.py     Experten-Datensatz: 4-8 Episoden, Live-SNN-Embedding,
                     Pulse-Frames, Ground-Truth-Würfelpositionen
step2_nmf.py         NMF-Rang 8 auf den Bewegungsresiduen → Prototypen-Bank
step3_kan_train.py   KAN als additive Splines (Ridge auf Hinge-Basen)
step4_kan_to_mlp.py  exakte ReLU-Kompilierung + numerische Verifikation
step5_train_heads.py Detektionskopf (Decoder) + Soft-MoE-Bias
step6_export.py      → weights.bin (Sektions-Blob)
```

## 1. Experten (step1)

Der Experte nutzt **Ground-Truth-Würfelpositionen** aus der Simulation —
er lernt schneller und reproduzierbar als jede perception-gespyste Variante.
Seine Phasenmaschine ist ereignisgetrieben (Phase endet, wenn der gemessene
TCP das Ziel erreicht, nicht wenn ein Timer abläuft). Pro Zyklus:

1. `mj_step × 5` (Physik)
2. Live-SNN: Event-Kamera → ALIF → Predictive Coding → 32-dim Embedding
3. Task-Maschine → tcp-Ziel → Closed-Form-IK → rate-limitiertes q_track
4. PD(+qfrc_bias)-Torques → ctrl

Datensatzzeilen (je Zyklus): `phase, s, q_start, q_goal, q_des, q, qd, grip,
grip_target, emb(32), pbins(576), cube_xy, cube_color, grasped`.

**Wichtig:** Die Rate-Limiter (1,5 rad/s) schützen die Szene vor IK-
Sprüngen — ohne sie schwingt der Arm beim Phasenwechsel durch die
Würfelreihe und schleudert Objekte vom Tisch.

## 2. NMF-Prototypen (step2)

Residuen `r(t) = q_des(t) − Basis(t)` pro Bewegungssegment (HOVER…PLACE,
HOME inklusive), auf 64 Samples resampelt, nichtnegative Hülle, MU-NMF
Rang 8. Die Bank P[8][7][64] ist die „Handschrift" des Experten: Handgelenk-
Vorbewegungen, Finger-Feinadjustierung. Die Ablage bleibt geometrisch
(siehe ARCHITEKTUR.md §4) — die Prototypen dürfen nötigenfalls versagen,
ohne die Aufgabe zu gefährden.

## 3. KAN-Training (step3) und Kompilierung (step4)

- **Knoten:** pro Eingang die Quantile 15/40/65/90 % der skalierten
  Trainingsdaten (degenerierte Spalten → Mikro-Spread).
- **Skalierung:** `in_scale` = Standardabweichung pro Eingang (One-Hots = 1).
- **Layer 1** (24→8): additive Splines direkt auf w* (Simplex-Ziele).
- **Layer 2** (8→8): additive Splines auf `log(w*)` (Logit-Raum).
- **Kompilierung:** die Hinge-Basis `[x ; ReLU(x−t)]` ist die identische
  Runtime-Form — `step4` prüft auf 2.000 Zufallseingaben, dass Runtime- und
  Modell-Forward übereinstimmen (Ergebnis: 0,0 Differenz).
- Runtime-Layout (`kan_mlp.cpp`):
  `h = W1ᵀ[24 ; 96 ReLU-Knicke] + b1`, `logits = W2ᵀ[8 ; 32 Knicke] + b2`.

## 4. Köpfe (step5)

- **Detektionskopf** (Decoder): Eingang `[embedding(32) ; gelatchte
  Pulse-Bins(576)]`, Ausgang 48 Werte = `x_off(8) , y(8) ,
  Farb-Logits(8×4)`. Ridge mit eingefalteter Standardisierung
  (`W/σ`, `b − (μ/σ)W`), sodass die Runtime ohne Vorverarbeitung auskommt.
  Val: **1,38 cm** mittlerer Positionsfehler.
- **Soft-MoE-Bias:** Ridge `emb → Δlogits`, wobei
  `Δlogits = log(w*) − logits_KAN` (trainiert die Lücke zwischen KAN und
  Expertengewichten). Zur Laufzeit wird der Bias auf die KAN-Logits addiert,
  bevor der Softmax die Prototypen-Mischung bildet.

## 5. Export (step6)

Sektions-Blob (little-endian, fp32):

```
PROTO   8·7·64      LSNN_W  576·128     LSNN_B  128        LSNN_G  128
PRED_FB 128·576     EMB_PROJ 704·32     SMOE_W  32·8       SMOE_B  8
KAN_W1  120·8       KAN_B1  8           KAN_W2  40·8        KAN_B2  8
HINGE_T1 24·4       HINGE_T2 16·4       DEC_W  608·48       DEC_B   48
LORA_A  8·4 (Null)  LORA_B  4·8 (Null)  IN_SCALE 24         META    8 floats
```

META = `[λ_v=0.85, ρ_a=0.92, β_θ=0.35, vth0=0.35, snn_η=0, pred_η=0.002,
out_scale=1, 0]`.

## 6. Nachtrainieren

1. `step1_collect.py N` mit größerem N → mehr Deckung (Ecken des Arbeitsraums).
2. Für mehr Greif-Robustheit: `GRIP_CAP`/`GRIP_CLOSED` in `types.h` +
   `common.py` variieren (Funnel- und Klemmdruck).
3. Pipeline 2–6 neu laufen lassen; `weights.bin` ins APK-Assets kopieren;
   APK neu bauen. Der C++-Loader validiert beim Start alle Sektionsgrößen —
   ein Mismatch fällt sofort auf.

## 7. Bekannte Grenzen

- Die Detektionskopf-Auflösung ist durch das 12×8-Raster begrenzt
  (~5 cm Zellgröße) — das Funnel-Design gleicht das aus; ein feineres
  Raster kostet LSNN-Zyklen.
- Farb-Logits lernen die Slot-feste Farbzuordnung (Szene-Konvention),
  nicht echte Farbperzeption — dafür wären rein farbige Events und
  mehr Episoden nötig.
- NMF-Restresiduen sind klein (0,037 rad): Die Prototypen sind Textur,
  kein Primärantrieb — so war es gedacht.
