"""Sweep completion models × X280 clock with the C++ bench, one process a point.

Appends `disp,<card>,<pll>,<case>,<model>,<us per command>,<ok|fail>` rows to a CSV.

Usage: TT_METAL_HOME=<tree> python bench/sweep_disp.py --card 0 --csv disp.csv
"""

import argparse
import os
import re
import subprocess

LINE = re.compile(r"^(?P<name>.+?)\s+slots=\s*(?P<slots>\d+)\s+(?P<us>[\d.]+) us/cmd")


def run_point(tree, card, pll, case, n):
    """One bench process; returns (model name, µs per command or '', 'ok' | 'fail')."""
    env = dict(
        os.environ,
        TT_METAL_RUNTIME_ROOT=tree,
        TT_VISIBLE_DEVICES=str(card),
        TT_METAL_SLOW_DISPATCH_MODE="1",
        TTX_N=str(n),
        TTX_DCASES=str(case),
        TTX_RSTRIDE="16",
        TTX_SPLIT="6",
    )
    bench = f"{tree}/build/test/tt_metal/x280/ttx_bench"
    proc = subprocess.run(
        [bench, "disp", str(pll)], env=env, capture_output=True, text=True, timeout=300
    )

    # The bench prints one result line per case, or a FAILED line.
    name, us, ok = "?", "", "fail"
    for line in proc.stdout.splitlines():
        match = LINE.match(line.strip())
        if match:
            name, us, ok = match["name"].strip(), match["us"], "ok"
        elif "FAILED" in line:
            name = line.split("FAILED")[0].strip()
    return name.replace(",", ";"), us, ok


def main():
    """Sweep the requested clocks and cases, appending one CSV row per run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--plls", type=int, nargs="+", default=[800, 1000, 1750])
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 14, 31, 21, 32])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    tree = os.environ["TT_METAL_HOME"]

    with open(args.csv, "a") as out:
        for pll in args.plls:
            for case in args.cases:
                for rep in range(args.reps):
                    name, us, ok = run_point(tree, args.card, pll, case, args.n)
                    out.write(f"disp,{args.card},{pll},{case},{name},{us},{ok}\n")
                    out.flush()
                    print(f"pll {pll} case {case} rep {rep}: {name} {us} {ok}")


if __name__ == "__main__":
    main()
