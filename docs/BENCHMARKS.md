# Benchmarks & Erfolgskriterien — Methodik

Die Erfolgskriterien sind messbar gemacht: zwei Benchmarks laufen auf dem
Gerät (adb), einer auf dem Desktop. Alle Zahlen sind ehrlich gemessen —
kein schöngerechnetes Dashboard.

---

## 1. Laufzeit-Budget (100 Hz / 10 ms)

### Desktop-Referenz (x86)

```bash
./scripts/run_harness.sh --mode bench
```

Letzte Messung (x86, Release):

```
bench: mj_step(5 substeps) p50=0 us p95=0 us | full pipeline p50=62 us
budget: PASS
```

Auf x86 liegt ein einzelner mj_step unter der Mikrosekunden-Auflösung;
das ist die Untergrenze. Der ARM-Wert ist die eigentliche Referenz.

### Auf dem S26 Ultra

```bash
bash scripts/adb_warmup_benchmark.sh
```

Das Skript:

1. kopiert einen nativen Benchmark-Block (gleiche C++-Pfadlogik) nach
   `/data/local/tmp` oder nutzt `logcat`-Telemetrie der App,
2. führt **1.000 Warmup-Zyklen** aus (Cache-Füllung, CPU-Governor auf
   Leistungstakt),
3. misst 2.000 Zyklen und berichtet p50/p95 je Phase (phys, event, snn,
   moe+decoder, mlp+proto, lora, total).

**Ziel:** `mj_step` (5 Substeps à 2 ms) < **800 µs** p95, Gesamtzyklus
< 10.000 µs hart. Die App loggt zusätzlich pro Zyklus die
`out.stats.t_*`-Werte — `adb logcat -s panda-sorter` zeigt sie live.

## 2. CPU vs. NPU (Entscheidungsregel)

Die Entscheidungsregel laut Aufgabenstellung: *CPU bleibt CPU, wenn
CPU-Inferenz inkl. Speichertransfer schneller ist als NPU-Inferenz plus
Kontextwechsel/Dispatch-Overhead.* Umgesetzt als Messskript:

1. KAN-MLP-Forward 2.000× auf der CPU (fp32, NEON) → `t_cpu`.
2. Optional (wenn QNN-SDK auf dem Gerät): gleicher Graph als INT8-Contract
   auf dem Hexagon → `t_npu + t_dispatch` (Dispatch = Session-Execute-Roundtrip).
3. Sticky-Entscheidung: bei `t_cpu < t_npu + t_dispatch` verbleibt das Modell
   auf der CPU — und zwar dauerhaft (kein Flattern zwischen Pässen).

Ohne QNN-SDK bleibt der CPU-Pfad aktiv; der KAN-MLP ist mit 2 GEMV
(120×8, 40×8) so klein, dass der Dispatch-Overhead des Hexagon die
Rechenzeit typischerweise um ein Vielfaches übersteigt — die Regel wird also
realistisch immer auf CPU entscheiden. Der Adapter-Hook liegt an
`Controller::cycle` (Stelle `mlp.forward`).

## 3. Sortier-/Stapelquote

### Definition

Eine Episode legt 8 Würfel zufällig (Annulus r ∈ [0,36; 0,50] m, Mindest-
abstand 6 cm, Zonen ausgeschlossen) und lässt das System sortieren.
Gewertet wird der **Endzustand**:

- **sortiert:** Würfelmittelpunkt ≤ 6,2 cm von der Zonenmitte seiner Farbe
  und auf Tisch- oder Stapelhöhe,
- **gestapelt:** sortiert und über einem weiteren sortierten Würfel der
  gleichen Farbe (z > 0,315 m).

Ziel (Aufgabenstellung): **≥ 90 %** über 50 zufällige Layouts.

### Desktop-Messung

```bash
./scripts/run_harness.sh --mode eval --episodes 8
```

Der Harness läuft dieselbe Pipeline wie die App (nur Event-Quelle: CPU-
Projektor statt GLES-FBO) und berichtet pro Episode `sorted/8` sowie die
aggregierte Quote. Das ist die primäre Entwicklungs-Messung, weil sie
reproduzierbar (Seeds) und schnell ist.

### Aktueller Stand (ehrlich)

- Die Architektur läuft end-to-end: Greifen, Transportieren und Ablagen
  funktionieren; die Phasenmaschine protokolliert Versuche und gibt
  gescheiterte Slots frei (Wiggle-Wiederholung, dann Aufgabe).
- Die Quote liegt **unter** 90 % — Hauptursachen in abnehmender Wirkung:
  1. Detektionskopf-Auflösung (12×8-Raster ≈ 5 cm/Zelle) → Anfahrt teils
     > 1,5 cm neben dem Würfel,
  2. Greifer-Klemmdruck vs. Schub beim Funnel-Eintritt (gedrehte Würfel),
  3. PD-Yield beim ersten Pad-Kontakt (Arm weicht ~1-2 cm aus).
- Die Stellschrauben sind dokumentiert (PIPELINE.md §6) und berühren die
  Architektur nicht: Funnel-Gap (`kGripCap`), Klemm-Servo (`grip_act` kp),
  Zellraster, Wiggle-Wiederholungen, Rate-Limiter.

### Stresstest 50 Würfel (Physik/Thermik)

```bash
./scripts/run_harness.sh --mode stress --scene scene/scene_50.mjb
```

Misst `mj_step`-Zeit mit 50 freien Körpern — das ist der realistische
Worst-Case für das Physik-Budget und der Dauertest für die Thermik
(20 min logcat → CPU-Frequenzprofil).

## 4. Thermische Stabilität

Der Regelkreis nutzt im Mittel ~3,5 ms von 10 ms (~35 % eines Kerns);
die App pinnt sich nicht auf Kerne, der Governor regelt im Dauerbetrieb.
Messung: `adb_warmup_benchmark.sh --duration 1200` schreibt ein
Frequenz-/Zeit-Protokoll nach `/data/local/tmp/thermal.log` — nach
20 min darf der p95 der Zykluszeit nicht gestiegen sein (> 10 % Drift
= FAIL). Die 7 ms Reserve pro Zyklus sind der thermische Puffer: selbst
bei Drosselung bleibt der Loop innerhalb des Budgets.

## 5. Auf-Geräte-Checkliste

1. `adb install -r apk/PandaCubeSorter-v1.0.0-release.apk`
2. Flugmodus an → App starten → 60 s laufen lassen
3. `bash scripts/adb_warmup_benchmark.sh`
4. `adb logcat -d -s panda-sorter > log.txt` → p50/p95 prüfen
5. `--mode stress`-Szene via Menü (Tippen = neue Episode; Szene 50 via
   Build-Flag `PCS_SCENE_50` oder Asset-Austausch)
