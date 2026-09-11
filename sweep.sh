#!/usr/bin/env bash
# The resolution study.
#
# The open question for the whole project: the LBM reports K = 12.5 for a duct
# that a correlation model, an independent hand calculation, and the measured
# geometry all put near K = 0.7. If that gap is under-resolution, K must fall
# as dx falls and head toward ~0.7. If it plateaus somewhere else, the cause is
# something other than resolution and this tells us that instead.
#
# Runs on the plenum domain, which is ~24x cheaper per step than the room and
# was measured to agree with it on dp at 350k steps (57 vs 60 Pa). That turns a
# 9.4-hour sweep into ~24 minutes.
#
# Strictly one run at a time: the solver is memory-bandwidth-bound, so a second
# GPU client moves every number.

set -uo pipefail
cd "$(dirname "$0")"

STEPS=500
FRAMES=700          # 350,000 steps -- generous at every dx here
OUT=sweep-out
mkdir -p "$OUT"

if tasklist 2>/dev/null | grep -qi 'aeroduct.exe'; then
    echo "refusing to start: an aeroduct process is already running"
    exit 1
fi

echo "resolution study: $((STEPS*FRAMES)) steps per point, plenum domain, 3 m/s"
echo

for dx in 0.75 0.5 0.4 0.3; do
    log="$OUT/dx$dx.log"
    printf '=== dx = %s mm ===\n' "$dx"
    start=$(date +%s)
    env AERODUCT_DOMAIN=plenum \
        AERODUCT_DX_MM="$dx" \
        AERODUCT_U=3 \
        AERODUCT_STEPS="$STEPS" \
        AERODUCT_FRAMES="$FRAMES" \
        AERODUCT_SHOT="$OUT/dx$dx.png" \
        ./target/release/aeroduct.exe > "$log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - start ))

    if [ $rc -ne 0 ]; then
        echo "  FAILED (exit $rc) after ${elapsed}s -- see $log"
        continue
    fi

    grep -oE 'grid [0-9]+ x [0-9]+ x [0-9]+|[0-9.]+ M cells' "$log" | head -2 | sed 's/^/  /'
    grep -E 'Q_in|dp_total|^ *K =|mass imbalance' "$log" | tail -4 | sed 's/^/  /'
    printf '  wall %ds\n\n' "$elapsed"
done

echo "=== summary: is K converging? ==="
for dx in 0.75 0.5 0.4 0.3; do
    k=$(grep -oE 'K = [0-9.]+ \+/- [0-9.]+' "$OUT/dx$dx.log" 2>/dev/null | tail -1)
    dp=$(grep -oE 'dp_total = [0-9]+ \+/- [0-9]+ Pa' "$OUT/dx$dx.log" 2>/dev/null | tail -1)
    printf '  dx=%-5s %-22s %s\n' "$dx" "${k:-no result}" "${dp:-}"
done
echo
echo "  correlation model says K = 0.71 +/- 0.14, dp = 3.8 Pa mouth-to-mouth."
