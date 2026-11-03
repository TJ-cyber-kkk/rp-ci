"""Norman Perturb-seq: Section 4.3.2, Table 14 and the consensus skeletons of Appendix D.8.

Writes table14.csv, norman_consensus.csv and norman_skeletons.csv to the output directory.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata

from rp_ci.gcm import gcm_test

from baselines import BASELINES
from dag import rpgcm

HERE = Path(__file__).resolve().parent
URL = "https://exampledata.scverse.org/scvi-tools/norman_2019_adata.h5ad"
MAIN_SEED, SUBSAMPLE_SEED = 20260501, 20260727  # fix the control cells of every graph fit
METHODS = ("rpgcm", "max_gcm", "mean_gcm", "kci", "rcit", "cmiknn")


def expression(data_dir):
    """log(1 + x) expression, x scaled to the median total over the 2,000 genes; cell barcodes; control cells."""
    import anndata

    path = Path(data_dir) / "norman_2019.h5ad"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    adata = anndata.read_h5ad(path)
    counts = adata.layers["count"]
    counts = (counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)).astype(np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    matrix = np.log1p(counts * (np.median(totals) / totals))
    labels = adata.obs["guide_identity"].astype(str)
    controls = np.flatnonzero([all(g.lower().startswith("negctrl") for g in v.replace("__", "_").split("_"))
                               for v in labels])
    return matrix, list(adata.obs_names), controls


def load(data_dir):
    """Node data, the control cells of each fit, and held-out support of every pair."""
    signatures = json.loads((HERE / "norman/01_SIGNATURES_EXACT.json").read_text())
    splits = json.loads((HERE / "norman/split_cells.json").read_text())
    names = list(signatures)
    matrix, barcodes, controls = expression(data_dir)
    fits = [np.sort(np.random.default_rng(seed).choice(controls, 1500, replace=False))
            for seed in [MAIN_SEED] + [SUBSAMPLE_SEED + i for i in range(10)]]
    reference = np.setdiff1d(controls, np.concatenate(fits))
    nodes = matrix[:, [c for name in names for c in signatures[name]["gene_cols"]]].reshape(len(matrix), len(names), 30)

    # Support of a -> b: mean absolute held-out response of b's genes to perturbing a's target.
    index = {cell: i for i, cell in enumerate(barcodes)}
    baseline = matrix[reference].mean(axis=0).astype(float)
    effects = {a: np.abs(matrix[[index[c] for c in splits[a]["eval"]]].mean(axis=0).astype(float) - baseline)
               for a in names}
    raw = np.array([[effects[a][signatures[b]["gene_cols"]].mean() for b in names] for a in names])
    normalized = np.zeros_like(raw)
    for a in range(len(names)):
        others = np.arange(len(names)) != a
        normalized[a, others] = rankdata(raw[a, others]) / (len(names) - 1)
    scores = {(i, j): (max(raw[i, j], raw[j, i]), max(normalized[i, j], normalized[j, i]))
              for i, j in itertools.combinations(range(len(names)), 2)}
    return nodes, names, fits, scores


def standardize(v):
    return (v - v.mean(axis=0)) / np.where(v.std(axis=0) < 1e-8, 1, v.std(axis=0))


def query(data, i, j, cond, method, seed, args):
    """p-value and dependence statistic of X_i independent of X_j given the nodes in cond."""
    x, y = data[i], data[j]
    z = np.concatenate([data[v] for v in cond], axis=1) if cond else np.empty((len(x), 0))
    if method == "rpgcm":
        p, stat = rpgcm(x, y, z, seed, k=args.k, return_statistic=True)
    elif method == "mean_gcm":
        zm = np.column_stack([data[v].mean(axis=1) for v in cond]) if cond else z
        result = gcm_test(x.mean(axis=1), y.mean(axis=1), zm)
        p, stat = result.pvalue, abs(result.statistic)
    else:
        p, stat = BASELINES[method](x, y, z, seed, return_statistic=True)
    return dict(i=i, j=j, cond=cond, pvalue=p, statistic=stat)


def pc(data, names, method, fit, args):
    """PC-stable to depth two, running every query of a depth before removing edges.

    Conditioning sets range over the union of the two endpoints' adjacencies.
    """
    adjacent = ~np.eye(len(names), dtype=bool)
    tests = []
    for depth in range(3):
        frozen = adjacent.copy()
        tasks = [(i, j, cond) for i, j in itertools.combinations(range(len(names)), 2) if frozen[i, j]
                 for cond in itertools.combinations(
                     sorted(set(np.flatnonzero(frozen[i] | frozen[j])) - {i, j}, key=lambda v: names[v]), depth)]
        seeds = [int(np.random.SeedSequence([args.seed, METHODS.index(method), fit, i, j, *cond]).generate_state(1)[0])
                 for i, j, cond in tasks]
        rows = Parallel(n_jobs=args.n_jobs)(delayed(query)(data, i, j, cond, method, seed, args)
                                            for (i, j, cond), seed in zip(tasks, seeds))
        tests += rows
        for row in rows:
            if row["pvalue"] > args.alpha:
                adjacent[row["i"], row["j"]] = adjacent[row["j"], row["i"]] = False
    return {(i, j) for i, j in itertools.combinations(range(len(names)), 2) if adjacent[i, j]}, tests


def rank_pairs(tests, names):
    """Largest p-value ascending; ties by the weakest statistic at that p-value, smallest p-value, name."""
    groups = defaultdict(list)
    for row in tests:
        groups[row["i"], row["j"]].append(row)
    keys = {}
    for pair, rows in groups.items():
        top = max(r["pvalue"] for r in rows)
        weakest = min(r["statistic"] for r in rows if r["pvalue"] == top)
        keys[pair] = (top, -weakest, min(r["pvalue"] for r in rows), "||".join(sorted(names[v] for v in pair)))
    return sorted(keys, key=keys.get)


def support(edges, scores, metric=1):
    """Mean held-out support of an edge set: metric 0 is raw, 1 is source-normalized."""
    return math.fsum(scores[e][metric] for e in edges) / len(edges)


def degree_null(edges, nodes, scores, seed, permutations, swaps_per_edge=5):
    """Support against degree-preserving edge swaps; dense graphs are swapped in their complement."""
    universe = set(itertools.combinations(range(nodes), 2))
    dense = len(edges) > len(universe) / 2
    current = universe - edges if dense else set(edges)
    target = len(current) * swaps_per_edge
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(permutations):
        done = 0
        for _ in range(max(100, 30 * target)):
            if len(current) < 2:
                break
            ordered = sorted(current)
            first, second = rng.choice(len(ordered), 2, replace=False)
            (a, b), (c, d) = ordered[first], ordered[second]
            if len({a, b, c, d}) < 4:
                continue
            new = ({(min(a, c), max(a, c)), (min(b, d), max(b, d))} if rng.random() < 0.5 else
                   {(min(a, d), max(a, d)), (min(b, c), max(b, c))})
            if not new & current:
                current -= {ordered[first], ordered[second]}
                current |= new
                done += 1
                if done == target:
                    break
        draws.append(support(universe - current if dense else current, scores))
    observed = support(edges, scores)
    return observed, float(np.mean(draws)), (1 + sum(v >= observed for v in draws)) / (permutations + 1)


def budget_rows(method, ranking, scores, budgets):
    return [dict(method=method, metric=metric, budget=budget, support=support(ranking[:budget], scores, index))
            for budget in budgets for metric, index in (("raw", 0), ("normalized", 1))]


def association_rankings(nodes, main, names):
    """Signature-mean correlation and partial correlation on the main control cells."""
    means = nodes[main].mean(axis=2)
    correlation = np.corrcoef(means, rowvar=False)
    precision = np.linalg.pinv(np.cov(means, rowvar=False))
    partial = -precision / np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    pairs = list(itertools.combinations(range(len(names)), 2))
    name = {p: "||".join(sorted(names[v] for v in p)) for p in pairs}
    return {method: sorted(pairs, key=lambda p: (-abs(matrix[p]), name[p]))
            for method, matrix in (("mean_correlation", correlation), ("mean_partial_correlation", partial))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 30, 60])
    ap.add_argument("--permutations", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    nodes, names, fits, scores = load(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, consensus, skeletons = [], [], []
    for method in args.methods:
        data = np.array([standardize(nodes[fits[0], v]) for v in range(len(names))])
        edges, tests = pc(data, names, method, 0, args)
        skeletons.append(dict(method=method, edges=len(edges)))
        table += budget_rows(method, rank_pairs(tests, names), scores, args.budgets)
        if method in ("rpgcm", "max_gcm"):
            sub = [pc(np.array([standardize(nodes[cells, v]) for v in range(len(names))]), names, method, f, args)[0]
                   for f, cells in enumerate(fits[1:], start=1)]
            counts = Counter(e for fit in sub for e in fit)
            kept = {e for e, c in counts.items() if c >= 6}
            sizes = [len(fit) for fit in sub]
            graphs = [("consensus", kept)] + ([("natural", edges)] if method == "rpgcm" else [])
            for label, graph in graphs:
                observed, mean, p = degree_null(graph, len(names), scores, [args.seed, METHODS.index(method)],
                                                args.permutations)
                spread = (np.mean(sizes), np.std(sizes, ddof=1)) if label == "consensus" else (None, None)
                consensus.append(dict(method=method, graph=label, edges=len(graph), mean_edges=spread[0],
                                      sd_edges=spread[1], support=observed, reference_mean=mean, p=p))
        print(method, "done", flush=True)
    for method, ranking in association_rankings(nodes, fits[0], names).items():
        table += budget_rows(method, ranking, scores, args.budgets)

    for name, rows in (("table14.csv", table), ("norman_consensus.csv", consensus), ("norman_skeletons.csv", skeletons)):
        if not rows:  # consensus is fitted only for rpgcm and max_gcm
            continue
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
