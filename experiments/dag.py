"""Known-DAG skeleton recovery with PC-stable (Tables 3 and 12) and matched nulls (Table 13).

Skeleton mode: random sparse 10-node DAGs with vector nodes in R^q and a linear
Gaussian SEM; every method runs PC-stable (depth <= 2) on the same datasets.
Matched-null mode (--matched-null): conditionally independent X, Y given s nodes.
"""
from __future__ import annotations

import argparse
import csv
import itertools

import numpy as np
from joblib import Parallel, delayed

from rp_ci import rp_test
from rp_ci.gcm import gcm_test, ols_residuals

from baselines import BASELINES

METHODS = ("rpgcm", "max_gcm", "cmiknn", "rcit", "kci")
MECHANISMS = ("full_rank", "rank_two")


def rpgcm(x, y, z, seed, k=1000, return_statistic=False):
    """Bonferroni p-value of RP-GCM with in-sample OLS residuals; the statistic is the max |T|."""
    # Residuals commute with projection, so the projected residuals need no further regression.
    rx, ry = ols_residuals(x, z), ols_residuals(y, z)
    statistics = []

    def projected_gcm(a, b, c):
        result = gcm_test(a, b, c)
        statistics.append(abs(result.statistic))
        return result.pvalue

    pvalue = min(1.0, k * rp_test(rx, ry, np.empty((len(x), 0)), projected_gcm, k=k, seed=seed).min_pvalue)
    return (pvalue, max(statistics)) if return_statistic else pvalue


TESTS = {"rpgcm": rpgcm, **BASELINES}


def random_dag(rng, nodes=10, p=0.24, max_parents=2, min_edges=5, max_edges=12):
    """Adjacency matrix (parent, child) in a random topological order."""
    while True:
        order = rng.permutation(nodes)
        directed = np.zeros((nodes, nodes), dtype=bool)
        for position in range(1, nodes):
            parents = order[:position][rng.random(position) < p]
            if len(parents) > max_parents:
                parents = rng.choice(parents, size=max_parents, replace=False)
            directed[parents, order[position]] = True
        if min_edges <= directed.sum() <= max_edges:
            return directed, order


def edge_map(rng, q, mechanism, strength=0.6):
    if mechanism == "full_rank":
        left = np.linalg.qr(rng.normal(size=(q, q)))[0]
        right = np.linalg.qr(rng.normal(size=(q, q)))[0]
        singular = strength * rng.uniform(0.35, 1.0, size=q)
        singular[-1] = strength
        return left @ np.diag(singular) @ right.T
    m = rng.normal(size=(q, 2)) @ rng.normal(size=(q, 2)).T
    return strength * m / np.linalg.norm(m, ord=2)


def vector_sem(directed, order, n, q, mechanism, rng):
    """Data of shape (nodes, n, q) from M_j = sum_{i in pa(j)} M_i W_ij + noise."""
    weights = {(i, j): edge_map(rng, q, mechanism) for i, j in zip(*np.nonzero(directed))}
    noise = rng.normal(size=(len(order), n, q))
    data = np.zeros_like(noise)
    for j in order:
        data[j] = noise[j]
        for i in np.flatnonzero(directed[:, j]):
            data[j] += data[i] @ weights[i, j]
    return data


def pc_stable_skeleton(data, test, alpha, max_depth, seed):
    """PC-stable: adjacencies are frozen within each depth, removals applied at its end."""
    nodes = data.shape[0]
    adjacent = ~np.eye(nodes, dtype=bool)
    for depth in range(max_depth + 1):
        frozen, removals, eligible = adjacent.copy(), [], False
        for i, j in itertools.combinations(range(nodes), 2):
            if not frozen[i, j]:
                continue
            sets = []
            for a, b in ((i, j), (j, i)):
                others = [v for v in np.flatnonzero(frozen[a]) if v != b]
                if len(others) >= depth:
                    eligible = True
                    sets += [tuple(sorted(s)) for s in itertools.combinations(others, depth)]
            for s in dict.fromkeys(sets):
                z = np.concatenate([data[v] for v in s], axis=1) if s else np.empty((data.shape[1], 0))
                query_seed = int(np.random.SeedSequence([seed, i, j, *s]).generate_state(1)[0])
                if test(data[i], data[j], z, query_seed) > alpha:
                    removals.append((i, j))
                    break
        if not eligible:
            break
        for i, j in removals:
            adjacent[i, j] = adjacent[j, i] = False
    return adjacent


def skeleton_run(graph, args):
    rng = np.random.default_rng([args.seed, graph])
    directed, order = random_dag(rng)
    truth = directed | directed.T
    upper = np.triu_indices(len(order), 1)
    rows = []
    for q, mechanism in itertools.product(args.q, args.mechanisms):
        data = vector_sem(directed, order, max(args.n), q, mechanism,
                          np.random.default_rng([args.seed, graph, q, MECHANISMS.index(mechanism)]))
        for n, method in itertools.product(args.n, args.methods):
            pc_seed = [args.seed, graph, q, n, MECHANISMS.index(mechanism), METHODS.index(method)]
            estimate = pc_stable_skeleton(data[:, :n], TESTS[method], args.alpha, 2,
                                          int(np.random.SeedSequence(pc_seed).generate_state(1)[0]))
            fp = int(np.sum(estimate[upper] & ~truth[upper]))
            fn = int(np.sum(~estimate[upper] & truth[upper]))
            rows.append([graph, n, q, mechanism, method, int(truth[upper].sum()), fp, fn, fp + fn])
    return rows


def matched_null(n, q, s, rep, method, args):
    """X, Y given s standard Gaussian nodes, with coefficients N(0, 0.45^2 / (s q))."""
    rng = np.random.default_rng([args.seed, n, q, s, rep])
    z = rng.normal(size=(n, s * q))
    bx, by = rng.normal(scale=0.45 / np.sqrt(max(s * q, 1)), size=(2, s * q, q))
    x, y = z @ bx + rng.normal(size=(n, q)), z @ by + rng.normal(size=(n, q))
    seed = int(np.random.SeedSequence([args.seed, n, q, s, rep, METHODS.index(method)]).generate_state(1)[0])
    return TESTS[method](x, y, z, seed) <= args.alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--n", nargs="+", type=int, default=[500, 1000])
    ap.add_argument("--q", nargs="+", type=int, default=[10, 30])
    ap.add_argument("--mechanisms", nargs="+", default=list(MECHANISMS), choices=MECHANISMS)
    ap.add_argument("--graphs", type=int, default=30)
    ap.add_argument("--matched-null", action="store_true")
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=1000, help="matched-null repetitions")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        if args.matched_null:
            writer.writerow(["n", "q", "s", "method", "replications", "rejects", "rate"])
            for n, q, s, method in itertools.product(args.n, args.q, args.depths, args.methods):
                rejects = sum(Parallel(n_jobs=args.n_jobs)(
                    delayed(matched_null)(n, q, s, rep, method, args) for rep in range(args.reps)))
                writer.writerow([n, q, s, method, args.reps, rejects, rejects / args.reps])
                f.flush()
                print(n, q, s, method, "done", flush=True)
        else:
            writer.writerow(["graph", "n", "q", "mechanism", "method", "true_edges", "fp", "fn", "shd"])
            for rows in Parallel(n_jobs=args.n_jobs, return_as="generator")(
                    delayed(skeleton_run)(g, args) for g in range(args.graphs)):
                writer.writerows(rows)
                f.flush()


if __name__ == "__main__":
    main()
