#!/usr/bin/env bash
# run_examples.sh <card> <out-dir> [mlp|flash_attn|resblock ...]: runs the Python examples
# on one card, one process each, appending one record per line to <out-dir>/examples.log
# and writing traces next to it. With no names, runs every example.
set -uo pipefail

card=${1:?usage: run_examples.sh <card> <out-dir> [example ...]}
out=${2:?usage: run_examples.sh <card> <out-dir> [example ...]}
shift 2
which=${*:-mlp flash_attn resblock}
mkdir -p "$out"
export TT_VISIBLE_DEVICES=$card

# One example run per process: a failed run cannot poison the next one.
run() {
    local example=$1
    shift
    timeout -s KILL 300 python -m "tt_x280.examples.$example" "$@" 2>&1 |
        grep -E "^\{|Error|timed out" >>"$out/examples.log"
}

for name in $which; do
    case $name in
    mlp)
        run mlp --shape 320 256 384 --trace "$out/mlp_320.pt"
        run mlp --shape 640 1024 768 --trace "$out/mlp_640.pt"
        run mlp --shape 640 1024 768 --host-fed
        ;;
    flash_attn)
        run flash_attn --shape 12 1280 64 --trace "$out/flash_attn.pt"
        run flash_attn --shape 8 2048 128
        run flash_attn --shape 16 512 64 --chunk 64 64
        run flash_attn --shape 12 1280 64 --exp-approx 0
        ;;
    resblock)
        run resblock --shape 1 56 56 64 --trace "$out/resblock.pt"
        run resblock --shape 2 28 28 128
        run resblock --shape 1 64 64 64
        run resblock --shape 1 56 56 64 --fidelity HiFi4
        run resblock --shape 2 28 28 128 --fidelity HiFi4
        ;;
    esac
done
