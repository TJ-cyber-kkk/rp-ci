"""Top-r variance diagnostic: Table 8.

For each null model and d, estimate on each dataset the fraction Q of fresh
projections whose p-value is at most the Beta-lambda threshold tau, and compare
the variance of Q across datasets with the margin m_r = 2 {alpha - g_r(tau)} / L_r,
where g_r(q) = P{Bin(k, q) >= r} and L_r = max_q [g_r''(q)]_+.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import beta, binom

from rp_ci import topr_threshold

from data_models import NULLS, generate
from synthetic import reference_pool, rpcit_pvalues


def margin(r, k, alpha, lam):
    tau = topr_threshold(alpha, k, r, rule="beta", lam=lam)
    if r == 2:
        L = float(k * (k - 1))  # g_2'' is maximal at q = 0.
    else:
        # Smaller stationary point of g_r'', the Beta(r, k-r+1) density derivative.
        a = r - 1
        q = (a * (k - 2) - np.sqrt(a * (k - 2) * (k - r))) / ((k - 1) * (k - 2))
        L = float(beta.pdf(q, r, k - r + 1) * ((r - 1) / q - (k - r) / (1 - q)))
    return tau, 2 * (alpha - binom.sf(r - 1, k, tau)) / L


def tail_fractions(model, d, dataset, taus, pool, args):
    ss = np.random.SeedSequence([args.seed, dataset, *f"{model}|{d}".encode()])
    data_ss, proj_ss = ss.spawn(2)
    x, y, z = generate(model, args.n, d, np.random.default_rng(data_ss))
    rng = np.random.default_rng(proj_ss)
    counts = np.zeros(len(taus))
    for start in range(0, args.M, 1000):
        p = rpcit_pvalues(x, y, z, pool, min(1000, args.M - start), rng)
        counts += [np.count_nonzero(p <= tau) for tau in taus]
    return counts / args.M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(NULLS), choices=NULLS)
    ap.add_argument("--d", nargs="+", type=int, default=[10, 50, 200])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--r", nargs="+", type=int, default=[2, 5, 10])
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--datasets", type=int, default=50)
    ap.add_argument("--M", type=int, default=100_000, help="projections per dataset")
    ap.add_argument("--B", type=int, default=500_000)
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--pool-dir", default="pools")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = reference_pool(args.n, 1, args.B, args.seed, args.pool_dir)
    constants = [margin(r, args.k, args.alpha, args.lam) for r in args.r]
    taus = [tau for tau, _ in constants]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "d", "r", "threshold", "margin", "s_B", "V_hat",
                         "ratio", "ratio_lo", "ratio_hi", "datasets", "M"])
        for model in args.models:
            for d in args.d:
                Q = np.array(Parallel(n_jobs=args.n_jobs)(
                    delayed(tail_fractions)(model, d, i, taus, pool, args)
                    for i in range(args.datasets)))
                boot = np.random.default_rng([args.seed, *f"{model}|{d}".encode()]).integers(
                    0, args.datasets, size=(args.bootstrap, args.datasets))
                for j, (r, (tau, m)) in enumerate(zip(args.r, constants)):
                    # Variance across datasets minus the projection sampling term.
                    q, qb = Q[:, j], Q[boot, j]
                    v = np.var(q, ddof=1) - np.mean(q * (1 - q)) / args.M
                    vb = np.var(qb, axis=1, ddof=1) - np.mean(qb * (1 - qb), axis=1) / args.M
                    mu = np.floor((args.B + 1) * tau) / (args.B + 1)
                    s_B = mu * (1 - mu) / (args.B + 2)  # Reference-pool term.
                    lo, hi = np.quantile((vb + s_B) / m, [0.025, 0.975])
                    writer.writerow([model, d, r, tau, m, s_B, v, (v + s_B) / m, lo, hi,
                                     args.datasets, args.M])
                f.flush()
                print(model, d, "done", flush=True)


if __name__ == "__main__":
    main()
