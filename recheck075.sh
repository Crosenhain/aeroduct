#!/usr/bin/env bash
# Is the dx = 0.75 mm anomaly real, and what causes it?
#
# K went 7.5 at 1.0 mm -> 10.1 at 0.75 mm -> 4.29 at 0.5 mm. A finer grid gave a
# worse answer, far outside the error bars. Four tests, each distinguishing a
# different explanation:
#
#   A  0.75 identical    - the solver is deterministic, so this must reproduce
#                          10.1 exactly. If it does not, either the harness is
#                          unsound or the submit-batching change altered the
#                          physics (it should not: the dispatch order and parity
#                          are unchanged, only where the command buffer is cut).
#   B  0.75 at 700k      - is 350k steps simply not converged in time? The room
#                          domain moved 65 -> 60 Pa between 70k and 350k, so
#                          temporal convergence is not established at 350k.
#   C  0.70 and 0.80     - is the anomaly specific to this dx, or a feature of
#                          the whole 0.7-0.8 range? A thin 6.3 mm passage
#                          staircases discretely, so the effective flow area can
#                          jump between neighbouring cell sizes.
set -uo pipefail
cd "$(dirname "$0")"
OUT=sweep-out; mkdir -p "$OUT"
run() {  # name dx steps frames
  printf '\n=== %s (dx=%s, %s steps) ===\n' "$1" "$2" "$(( $3 * $4 ))"
  s=$(date +%s)
  env AERODUCT_DOMAIN=plenum AERODUCT_DX_MM="$2" AERODUCT_U=3 \
      AERODUCT_STEPS="$3" AERODUCT_FRAMES="$4" \
      ./target/release/aeroduct.exe > "$OUT/$1.log" 2>&1
  printf '  exit %d, wall %ds\n' "$?" "$(( $(date +%s) - s ))"
  grep -oE 'K = [0-9.]+ \+/- [0-9.]+|dp_total = [0-9.]+ \+/- [0-9.]+ Pa|mass imbalance = [0-9.]+%|Q_in = [0-9.]+ \+/- [0-9.]+ L/s' \
      "$OUT/$1.log" 2>/dev/null | tail -4 | sed 's/^/  /'
}
until ! tasklist 2>/dev/null | grep -qi 'aeroduct.exe'; do sleep 15; done
run recheck-0.75-same 0.75 500 700
run recheck-0.75-long 0.75 500 1400
run recheck-0.70      0.70 500 700
run recheck-0.80      0.80 500 700
echo
echo "=== against the original 0.75 run: K = 10.1 +/- 0.4, dp = 56 +/- 3 Pa ==="
echo "RECHECK DONE"
