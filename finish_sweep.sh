#!/usr/bin/env bash
# Finish the resolution study: the two fine points the session ran out of time on.
set -uo pipefail
cd "$(dirname "$0")"
# 0.25 mm removed deliberately. After 0.3 mm the sweep has four points
# (0.75/0.5/0.4/0.3), which is one degree of freedom against the three fit
# parameters and therefore the first point at which the convergence order can be
# tested rather than assumed. A fifth point at 0.25 has a refinement ratio of
# only 1.20 -- the weakest in the sweep -- which amplifies measurement noise by
# 3.3x in the extrapolation, and costs ~101 minutes. Low leverage, high price.
for dx in 0.3; do
    printf '=== dx = %s mm ===\n' "$dx"
    s=$(date +%s)
    env AERODUCT_DOMAIN=plenum AERODUCT_DX_MM="$dx" AERODUCT_U=3 \
        AERODUCT_STEPS=500 AERODUCT_FRAMES=700 \
        AERODUCT_SHOT="sweep-out/dx$dx.png" \
        ./target/release/aeroduct.exe > "sweep-out/dx$dx.log" 2>&1
    rc=$?
    printf '  exit %d, wall %ds\n' "$rc" "$(( $(date +%s) - s ))"
    grep -E 'M cells|Q_in =|dp_total|^ *K =|mass imbalance' "sweep-out/dx$dx.log" | tail -5 | sed 's/^/  /'
    echo
done
echo "FINE SWEEP DONE"
