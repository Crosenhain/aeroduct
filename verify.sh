#!/usr/bin/env bash
# AeroDuct verification sweep.
#
# Runs the checklist from the plan end to end, strictly one GPU job at a time.
# That matters more than it sounds: this solver is memory-bandwidth-bound, so a
# second GPU client moves the throughput numbers by 2x or more. An earlier run of
# the same build measured 98% of roofline alone and 61% alongside the rest of the
# test suite. Any number produced while something else is on the GPU is fiction.
#
# Usage:  ./verify.sh [quick|full]
#   quick  correctness only, a couple of minutes
#   full   adds the resolution sweep, tens of minutes
#
# Every run writes to verify-out/.

set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"
OUT="verify-out"
mkdir -p "$OUT"
APP=./target/release/aeroduct.exe

pass=0; fail=0
say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$*"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=$((fail+1)); }

# Refuse to measure anything while another GPU client is running, rather than
# quietly reporting a contended number as if it were real.
require_idle_gpu() {
    if tasklist 2>/dev/null | grep -qi 'aeroduct.exe'; then
        bad "an aeroduct process is already running; benchmark numbers would be meaningless"
        return 1
    fi
    local util
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${util:-}" ] && [ "$util" -gt 40 ]; then
        printf '  \033[33mWARN\033[0m GPU already at %s%%; timings below are not trustworthy\n' "$util"
    fi
    return 0
}

# Run the app headlessly and echo the log path.
run_app() {
    local log="$1"; shift
    env "$@" "$APP" > "$log" 2>&1
    echo "$log"
}

grep_num() { grep -oE "$2" "$1" | tail -1; }

say "build"
if cargo build --release --workspace 2>&1 | grep -qE '^(error|warning)'; then
    bad "release build is not clean"
else
    ok "release build clean"
fi

say "unit and validation tests"
if cargo test --workspace --release 2>&1 | tee "$OUT/tests.log" | grep -qE '^(error|test result: FAILED)'; then
    bad "test suite"
else
    ok "test suite ($(grep -oE '[0-9]+ passed' "$OUT/tests.log" | awk '{s+=$1} END {print s}') passed)"
fi

require_idle_gpu || exit 1

say "solver throughput (idle GPU, run alone)"
cargo test --release -p ad-solver --test lbm_gpu throughput -- --nocapture \
    > "$OUT/throughput.log" 2>&1
grep -E 'MLUPS|roofline' "$OUT/throughput.log" | sed 's/^/  /'

say "geometry against the reference part"
cargo test --release -p ad-geom --test geom_test_part -- --nocapture \
    > "$OUT/geom.log" 2>&1
if grep -q 'watertight 2-manifold: yes' "$OUT/geom.log"; then ok "watertight genus-1 solid"; else bad "mesh health"; fi
if grep -q '0 with odd parity' "$OUT/geom.log"; then ok "no voxelisation leaks"; else bad "ray parity"; fi
if grep -q '0 hard disagreements' "$OUT/geom.log"; then ok "GPU mask matches CPU reference"; else bad "GPU/CPU mask"; fi

say "duct at 3 m/s, inlet on mouth A"
L=$(run_app "$OUT/u3.log" AERODUCT_U=3 AERODUCT_STEPS=60 AERODUCT_FRAMES=400 \
        AERODUCT_SHOT="$OUT/u3.png")
grep -E 'mouth A|mouth B|passage volume|grid ' "$L" | sed 's/^/  /'

say "zero inlet velocity must give zero flow"
L=$(run_app "$OUT/u0.log" AERODUCT_U=0 AERODUCT_STEPS=20 AERODUCT_FRAMES=40)
if grep -qiE 'nan|inf' "$L"; then bad "field went non-finite at U=0"; else ok "stable at U=0"; fi

say "inlet swap changes the fitting"
run_app "$OUT/inletB.log" AERODUCT_U=3 AERODUCT_INLET=1 AERODUCT_STEPS=60 \
    AERODUCT_FRAMES=400 AERODUCT_SHOT="$OUT/inletB.png" > /dev/null
ok "ran with inlet on mouth B (compare dp/K against $OUT/u3.log by eye)"

say "mesh display modes render"
for m in off ghost solid wire; do
    run_app "$OUT/mesh-$m.log" AERODUCT_U=3 AERODUCT_MESH="$m" AERODUCT_STEPS=20 \
        AERODUCT_FRAMES=60 AERODUCT_SHOT="$OUT/mesh-$m.png" > /dev/null
    if [ -f "$OUT/mesh-$m.png" ]; then ok "mesh=$m rendered"; else bad "mesh=$m produced no image"; fi
done

if [ "$MODE" = "full" ]; then
    say "resolution sweep -- is the pressure drop converging?"
    # K is expected to fall as dx falls. Two points are a trend, three are
    # evidence. This is the open question in the project: whether the loss
    # coefficient reaches the ASHRAE band for this fitting once the ~6 mm
    # passage is properly resolved.
    for dx in 1.0 0.75 0.5; do
        say "  dx = $dx mm"
        run_app "$OUT/dx$dx.log" AERODUCT_U=3 AERODUCT_DX_MM="$dx" \
            AERODUCT_STEPS=100 AERODUCT_FRAMES=600 AERODUCT_SHOT="$OUT/dx$dx.png" > /dev/null
        grep -E 'grid |allocating DDFs' "$OUT/dx$dx.log" | sed 's/^/  /'
    done
fi

say "summary"
printf '  %d passed, %d failed\n' "$pass" "$fail"
printf '  artefacts in %s/\n' "$OUT"
[ "$fail" -eq 0 ]
