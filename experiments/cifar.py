"""CIFAR-10 patches: Figure 2 (Appendix D.5).

One CSV row per (method, n, eta) with rejection counts for all five methods.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from rp_ci import aggregate

from baselines import BASELINES
from synthetic import reference_pool, rpcit_pvalues

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
BATCHES = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
METHODS = ("rpcit", "kci", "cmiknn", "rcit", "max_gcm")


def load_images(data_dir):
    """All 60,000 training and test images as uint8 arrays of shape (32, 32, 3)."""
    archive = Path(data_dir) / "cifar-10-python.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        batches = [pickle.load(tar.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")[b"data"]
                   for name in BATCHES]
    return np.concatenate(batches).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def make_data(images, n, eta, rng):
    """X: upper-left patch of image a; Y: lower-right patch of a with probability eta, else of b != a."""
    a = rng.integers(len(images), size=n)
    b = rng.integers(len(images), size=n)
    while np.any(same := b == a):
        b[same] = rng.integers(len(images), size=same.sum())
    z = rng.uniform(-1, 1, size=(n, 1))
    source = np.where(rng.random(n) < eta, a, b)
    scale = (1 + 0.4 * z)[:, :, None, None]

    def patch(index, corner):
        bright = np.clip(images[index][:, corner, corner] / np.float32(255) * scale, 0, 1)
        return np.clip(bright + rng.normal(0, 0.02, bright.shape), 0, 1).reshape(n, -1)

    return patch(a, slice(0, 16)), patch(source, slice(16, 32)), z


def replication(images, n, eta, rep, args, pool):
    """Apply each method to one dataset; only RP-CIT runs beyond args.reps at eta = 0."""
    data_ss, proj_ss, method_ss = np.random.SeedSequence([args.seed, n, round(1000 * eta), rep]).spawn(3)
    x, y, z = make_data(images, n, eta, np.random.default_rng(data_ss))
    seed = int(method_ss.generate_state(1)[0])
    out = {}
    for method in args.methods:
        if rep >= (args.null_reps if method == "rpcit" and eta == 0 else args.reps):
            continue
        if method == "rpcit":
            p = rpcit_pvalues(x, y, z, pool, args.k, np.random.default_rng(proj_ss))
            out[method] = aggregate(p, alpha=args.alpha).reject
        else:
            out[method] = BASELINES[method](x, y, z, seed) <= args.alpha
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--n", nargs="+", type=int, default=[100, 200, 500])
    ap.add_argument("--eta", nargs="+", type=float, default=[0, 0.1, 0.25, 0.5, 0.75, 1])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--null-reps", type=int, default=1000, help="RP-CIT repetitions at eta = 0")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=1_000_000, help="reference pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pool-dir", default="pools")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images = load_images(args.data_dir)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n", "eta", "replications", "rejects", "rate"])
        for n in args.n:
            pool = reference_pool(n, 1, args.B, args.seed, args.pool_dir) if "rpcit" in args.methods else None
            for eta in args.eta:
                count = max(args.reps, args.null_reps) if eta == 0 and "rpcit" in args.methods else args.reps
                results = Parallel(n_jobs=args.n_jobs)(
                    delayed(replication)(images, n, eta, rep, args, pool) for rep in range(count))
                for method in args.methods:
                    decisions = [r[method] for r in results if method in r]
                    writer.writerow([method, n, eta, len(decisions), sum(decisions), sum(decisions) / len(decisions)])
                f.flush()
                print(n, eta, "done", flush=True)


if __name__ == "__main__":
    main()
