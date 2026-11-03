"""RP-GCM and Max-GCM on the same datasets: Table 11.

Panel (a): --n 2000 --dz 1 5 10 50.  Panel (b): --models N1 N2 --n 500 1000 2000 5000 --dz 50.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from joblib import Parallel, delayed

from baselines import max_gcm
from dag import rpgcm
from data_models import GCM_MODELS, generate_gcm


def replication(model, n, dz, rep, args):
    ss = np.random.SeedSequence([args.seed, rep, n, dz, *model.encode()])
    data_seed, proj_seed, gcm_seed = (int(s) for s in ss.generate_state(3))
    x, y, z = generate_gcm(model, n, dz, np.random.default_rng(data_seed))
    return (rpgcm(x, y, z, proj_seed, k=args.k) <= args.alpha,
            max_gcm(x, y, z, gcm_seed) <= args.alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(GCM_MODELS), choices=GCM_MODELS)
    ap.add_argument("--n", nargs="+", type=int, default=[2000])
    ap.add_argument("--dz", nargs="+", type=int, default=[1, 5, 10, 50])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "dz", "method", "replications", "rejects", "rate"])
        for n in args.n:
            for dz in args.dz:
                for model in args.models:
                    results = np.array(Parallel(n_jobs=args.n_jobs)(
                        delayed(replication)(model, n, dz, rep, args)
                        for rep in range(args.reps)))
                    for method, rejects in zip(("rpgcm", "max_gcm"), results.sum(axis=0)):
                        writer.writerow([model, n, dz, method, args.reps, int(rejects),
                                         rejects / args.reps])
                    f.flush()
                    print(model, n, dz, "done", flush=True)


if __name__ == "__main__":
    main()
