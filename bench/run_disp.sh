#!/usr/bin/env bash
# run_disp.sh <card> <case> <reps> [ENV=VAL ...]: one protocol-bench process per rep;
# prints the result and the per-core post-mortem. Needs TT_METAL_HOME (the built tree).
set -u

card=$1
case_id=$2
reps=$3
shift 3
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal tree}"
log_dir=${LOG_DIR:-$PWD/logs}
mkdir -p "$log_dir"

for r in $(seq 1 "$reps"); do
    log=$log_dir/disp_c${card}_k${case_id}_$(date +%H%M%S)_$r.log

    # One process per run: a failed case cannot poison the next one.
    (
        cd "$TT_METAL_HOME" &&
            env TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME" TT_VISIBLE_DEVICES="$card" \
                TT_METAL_SLOW_DISPATCH_MODE=1 TTX_N="${TTX_N:-20000}" \
                TTX_DCASES="$case_id" TTX_SPLIT="${TTX_SPLIT:-6}" "$@" \
                timeout 300 build/test/tt_metal/x280/ttx_bench disp "${TTX_PLL:-1750}"
    ) >"$log" 2>&1
    rc=$?

    echo "[card $card case $case_id rep $r rc=$rc $*]"
    grep -E "us/cmd|FAILED|group acks|core \(|diag totals|not ready|what\(\)" "$log" | cut -c1-180
done
