"""Synthetic benchmarks: Tables 1, 2, 4, 5, 6, 9, 10 and Figure 1.

Each replication draws one dataset and applies every requested method to it.
Output: one CSV row per (model, Z, d, n, method, rule, r, k) with rejection counts.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from rp_ci import aggregate
from rp_ci.cit import (direct_cit_statistic_fast, plus_one_pvalues,
                       scalar_projected_statistics, simulate_null_pool)
from rp_ci.projections import batch_unit_random_projection

from baselines import BASELINES
from data_models import MODELS, generate

H_SCALE = 0.8


def reference_pool(n, d, B, seed, pool_dir):
    """Sorted pool of B null statistics for CIT with (dx, dy, dz) = (d, d, 1)."""
    path = Path(pool_dir) / f"pool_n{n}_d{d}_B{B}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    pool = np.sort(simulate_null_pool(n, d, d, 1, B=B, seed=seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pool)
    return pool


def rpcit_pvalues(x, y, z, pool, k, rng):
    """Plus-one p-values of k projected CIT statistics, sharing one reference pool."""
    xp = batch_unit_random_projection(x, rng, k)
    yp = batch_unit_random_projection(y, rng, k)
    stats = scalar_projected_statistics(xp, yp, z, h_scale=H_SCALE)
    return plus_one_pvalues(stats, pool, assume_sorted=True)


def replication(setting, rep, args, pools):
    model, z_dist, d, n = setting
    ss = np.random.SeedSequence([args.seed, rep, *f"{model}|{z_dist}|{d}|{n}".encode()])
    data_ss, proj_ss = ss.spawn(2)
    method_seed = int(ss.generate_state(1)[0])
    x, y, z = generate(model, n, d, np.random.default_rng(data_ss), z_dist)
    out = {}
    if "rpcit" in args.methods:
        p = rpcit_pvalues(x, y, z, pools[(n, 1)], args.k, np.random.default_rng(proj_ss))
        for k in args.k_prefixes or [args.k]:
            out["rpcit", "max", 1, k] = aggregate(p[:k], alpha=args.alpha).reject
        for rule in args.topr:
            for r in args.r:
                res = aggregate(p, alpha=args.alpha, aggregation=rule, r=r, lam=args.lam)
                out["rpcit", rule, r, args.k] = res.reject
    if "direct" in args.methods:
        statistic = direct_cit_statistic_fast(x, y, z, h_scale=H_SCALE)[0]
        out["direct", "", "", ""] = plus_one_pvalues(statistic, pools[(n, d)], assume_sorted=True) <= args.alpha
    for name in args.methods:
        if name in BASELINES:
            out[name, "", "", ""] = BASELINES[name](x, y, z, method_seed) <= args.alpha
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--d", nargs="+", type=int, default=[10, 50, 200])
    ap.add_argument("--n", nargs="+", type=int, default=[100])
    ap.add_argument("--z", default="gaussian", choices=("gaussian", "bounded"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--methods", nargs="+", default=["rpcit"],
                    choices=["rpcit", "direct", *BASELINES])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--k-prefixes", nargs="+", type=int,
                    help="also apply max aggregation to the first k projections")
    ap.add_argument("--topr", nargs="*", default=[], choices=("beta", "markov"))
    ap.add_argument("--r", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=500_000, help="reference pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--pool-dir", default="pools")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.k_prefixes and max(args.k_prefixes) > args.k:
        ap.error("--k-prefixes must not exceed --k")

    settings = [(m, args.z, d, n) for n in args.n for d in args.d for m in args.models]
    pools = {}
    for n in args.n:
        if "rpcit" in args.methods:
            pools[n, 1] = reference_pool(n, 1, args.B, args.seed, args.pool_dir)
        if "direct" in args.methods:
            for d in args.d:
                pools[n, d] = reference_pool(n, d, args.B, args.seed, args.pool_dir)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "z", "d", "n", "method", "rule", "r", "k",
                         "replications", "rejects", "rate"])
        for setting in settings:
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(replication)(setting, rep, args, pools) for rep in range(args.reps))
            for key in results[0]:
                rejects = sum(bool(res[key]) for res in results)
                writer.writerow([*setting, *key, args.reps, rejects, rejects / args.reps])
            f.flush()
            print(*setting, "done", flush=True)


if __name__ == "__main__":
    main()
