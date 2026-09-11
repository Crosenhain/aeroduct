#!/usr/bin/env bash
# Extend the resolution study onto the coarse end: 1 mm and 2 mm.
#
# Cheap (2.7 M and 0.34 M cells, minutes each) and they buy two things:
#
#   1. Range. With 0.3-2.0 mm the sweep spans 6.7x in cell size, which is far
#      more leverage for identifying the convergence order than the 0.3-0.75
#      cluster alone.
#   2. A number for the real-time tier. The computed real-time barrier for this
#      domain is dx = 2.56 mm, so 2 mm is essentially "what a genuinely
#      real-time answer would say". The passage is 6.3 mm, i.e. ~3 cells across
#      at 2 mm, so this quantifies how wrong real-time would be rather than
#      leaving it as an assertion.
#
# Waits for the running 0.3 mm point first, and cancels the queued 0.25 mm run
# (deliberately dropped: ratio 1.20 is the weakest in the sweep and it costs
# ~101 min). Strictly one GPU job at a time throughout.

set -uo pipefail
cd "$(dirname "$0")"

OUT=sweep-out
mkdir -p "$OUT"

echo "waiting for the 0.3 mm run to finish..."
while ! grep -q "metrics at step" "$OUT/dx0.3.log" 2>/dev/null; do
    # If nothing is running and the log never got results, the run died.
    if ! tasklist 2>/dev/null | grep -qi 'aeroduct.exe'; then
        if ! grep -q "metrics at step" "$OUT/dx0.3.log" 2>/dev/null; then
            echo "0.3 mm exited without writing results; continuing to the coarse points"
            break
        fi
    fi
    sleep 30
done
echo "0.3 mm done (or gone)"

# The already-running sweep script may still start 0.25 mm. Editing a script
# under a live bash is unreliable, so cancel it here instead.
for _ in $(seq 1 20); do
    sleep 10
    if [ -f "$OUT/dx0.25.log" ]; then
        echo "cancelling the queued 0.25 mm run"
        taskkill //F //IM aeroduct.exe 2>&1 | head -1
        rm -f "$OUT/dx0.25.log"
        sleep 5
        break
    fi
    tasklist 2>/dev/null | grep -qi aeroduct || break
done

# Make sure the GPU really is ours before timing anything.
until ! tasklist 2>/dev/null | grep -qi 'aeroduct.exe'; do sleep 15; done

for dx in 1.0 2.0; do
    printf '\n=== dx = %s mm ===\n' "$dx"
    s=$(date +%s)
    env AERODUCT_DOMAIN=plenum AERODUCT_DX_MM="$dx" AERODUCT_U=3 \
        AERODUCT_STEPS=500 AERODUCT_FRAMES=700 \
        AERODUCT_SHOT="$OUT/dx$dx.png" \
        ./target/release/aeroduct.exe > "$OUT/dx$dx.log" 2>&1
    rc=$?
    printf '  exit %d, wall %ds\n' "$rc" "$(( $(date +%s) - s ))"
    grep -E 'M cells|grid |Q_in =|dp_total|^ *K =|mass imbalance|cells across' \
        "$OUT/dx$dx.log" 2>/dev/null | tail -6 | sed 's/^/  /'
done

echo
echo "=== full sweep, coarse to fine ==="
for dx in 2.0 1.0 0.75 0.5 0.4 0.3; do
    k=$(grep -oE 'K = [0-9.]+ \+/- [0-9.]+' "$OUT/dx$dx.log" 2>/dev/null | tail -1)
    dp=$(grep -oE 'dp_total = [0-9.]+ \+/- [0-9.]+ Pa' "$OUT/dx$dx.log" 2>/dev/null | tail -1)
    printf '  dx=%-5s %-24s %s\n' "$dx" "${k:-(no result)}" "${dp:-}"
done
echo
echo "  correlation model: K = 0.71 +/- 0.14, dp = 3.8 Pa mouth-to-mouth"
echo "COARSE SWEEP DONE"
