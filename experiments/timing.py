"""Single-core runtime of RP-CIT on N1 with Gaussian Z: Table 7.

Each timing covers projection, projected CIT statistics, plus-one p-values and
max aggregation; data generation and pool simulation are excluded.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"  # Must precede the NumPy import.

import argparse
import csv
import time

import numpy as np

from data_models import generate
from rp_ci import aggregate
from synthetic import reference_pool, rpcit_pvalues

SETTINGS = ([("vary_k", 100, 50, k) for k in (100, 500, 1000, 2000)]
            + [("vary_n", n, 50, 1000) for n in (100, 200, 500)]
            + [("vary_d", 100, d, 1000) for d in (50, 200, 500)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--B", type=int, default=1_000_000,
                    help="reference pool size; it barely affects the timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-dir", default="pools")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "n", "d", "k", "runs", "median_seconds",
                         "q25_seconds", "q75_seconds"])
        for index, (name, n, d, k) in enumerate(SETTINGS):
            pool = reference_pool(n, 1, args.B, args.seed, args.pool_dir)
            seconds = []
            for run in range(args.runs + 1):  # The first run is a warm-up.
                data_ss, proj_ss = np.random.SeedSequence([args.seed, index, run]).spawn(2)
                x, y, z = generate("N1", n, d, np.random.default_rng(data_ss))
                start = time.perf_counter()
                p = rpcit_pvalues(x, y, z, pool, k, np.random.default_rng(proj_ss))
                aggregate(p, alpha=0.05)
                seconds.append(time.perf_counter() - start)
            q25, median, q75 = np.quantile(seconds[1:], [0.25, 0.5, 0.75])
            writer.writerow([name, n, d, k, args.runs, median, q25, q75])
            f.flush()
            print(name, n, d, k, f"{median:.3f}s", flush=True)


if __name__ == "__main__":
    main()
