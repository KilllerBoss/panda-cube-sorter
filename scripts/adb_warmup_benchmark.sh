#!/usr/bin/env bash
# adb_warmup_benchmark.sh — Warmup + Zyklus-Timing auf dem S26 Ultra
# Voraussetzung: App installiert, USB-Debugging aktiv.
# Die App loggt pro Zyklus die Phasenzeiten; dieses Skript sammelt sie und
# berechnet p50/p95 je Phase über das Messfenster.
set -e
DURATION=${1:-60}   # Messfenster in Sekunden (default 60; 1200 = 20-min-Thermik)

echo "== panda-cube-sorter on-device benchmark =="
echo "== Phase 1: Warmup (App 30 s laufen lassen, Cache/Governor) =="
adb logcat -c
adb shell monkey -p dev.pandasorter -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1 || true
sleep 30

echo "== Phase 2: Messung (${DURATION}s) =="
adb logcat -d -s panda-sorter:V > /tmp_pcs_log.txt 2>/dev/null || adb logcat -d -s panda-sorter:V > pcs_log.txt
LOG=pcs_log.txt; [ -f /tmp_pcs_log.txt ] && LOG=/tmp_pcs_log.txt
sleep "$DURATION"
adb logcat -d -s panda-sorter:V > "$LOG"

echo "== Auswertung =="
# Die App schreibt Zeilen wie: "cyc <n> phys=..us snn=..us mlp=..us total=..us"
for phase in phys snn mlp total; do
  grep -o "${phase}=[0-9]*" "$LOG" | cut -d= -f2 | sort -n > /tmp_pcs_${phase}.txt 2>/dev/null || true
  n=$(wc -l < /tmp_pcs_${phase}.txt 2>/dev/null || echo 0)
  if [ "$n" -gt 10 ]; then
    p50=$(sed -n "$((n/2))p" /tmp_pcs_${phase}.txt)
    p95=$(sed -n "$((n*95/100))p" /tmp_pcs_${phase}.txt)
    echo "  $phase: p50=${p50}us p95=${p95}us (n=$n)"
  fi
done
rm -f /tmp_pcs_*.txt

echo "== Budget-Check: total p95 < 10000us, phys p95 < 800us =="
echo "== Thermik: mit DURATION=1200 wiederholen und Drift vergleichen =="
