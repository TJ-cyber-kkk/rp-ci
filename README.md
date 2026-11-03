# RP-CI

Random-projection aggregation for conditional independence testing: project
high-dimensional `X` and `Y` to scalars, leave `Z` unprojected, and combine
user-supplied base-test p-values using Bonferroni or top-r aggregation. CIT and
GCM base tests are included.

## Installation

Requires Python 3.10 or later. From the repository root:

```bash
python -m pip install -e .
```

## Usage

Given data arrays `X`, `Y`, `Z` and a compatible `scalar_ci_test`:

```python
from rp_ci import rp_test

result = rp_test(
    X, Y, Z, base_test=scalar_ci_test, k=100, alpha=0.05, seed=42
)
print(result.reject)
```

For a runnable interface demo with a placeholder base test, run
`python examples/quickstart.py`.

Inputs must have matching row counts; one-dimensional inputs are treated as
columns. `scalar_ci_test(x, y, z)` receives float arrays of shapes `(n, 1)`,
`(n, 1)`, and `(n, d_z)`. It must return a finite scalar p-value in `[0, 1]`
without modifying inputs. `seed` controls projections only.

| Field | Meaning |
| --- | --- |
| `reject` | Whether `rth_pvalue <= threshold` |
| `min_pvalue` | Smallest projection-wise p-value, without adjustment |
| `rth_pvalue` | `r`-th smallest projection-wise p-value; `min_pvalue` for max |
| `threshold` | Threshold of the aggregation rule; `alpha / k` for max |
| `rule`, `r` | Aggregation rule and `r` (`"max"` and `1` by default) |
| `pvalues` | All projection-wise p-values, shape `(k,)` |

Calibration belongs to the base test: a CIT adapter uses plus-one p-values and
reuses one pool of `B` independent reference statistics, generated independently
of the data and projections.

Type I control depends on valid base-test p-values, not independence between
them. For asymptotically valid base tests, the guarantee is for fixed `k`.

## Top-r aggregation

Top-r aggregation rejects when the `r`-th smallest p-value is at most a
threshold, which can help when the signal is spread over several projections:

```python
result = rp_test(
    X, Y, Z, base_test=scalar_ci_test, k=1000, alpha=0.05, seed=42,
    aggregation="beta", r=10,
)
```

| `aggregation` | Threshold |
| --- | --- |
| `"max"` (default) | `alpha / k` |
| `"markov"` | `alpha * r / k` |
| `"beta"` | `alpha`-quantile of `Beta(r, k - r + 1)`, divided by `lam` (default `1.5`) |

`"markov"` has the same guarantee as `"max"` but is conservative. `"beta"` is
sharper but additionally requires the null probability that a projection's
p-value falls below the threshold, conditional on the sample, to have small
variance; a larger `lam` allows a larger variance.
`aggregate(pvalues, alpha=..., aggregation=..., r=...)` applies a rule to
precomputed projection p-values, and `topr_threshold(alpha, k, r)` returns the
top-r threshold.

## Base tests

- `rp_ci.cit.cit_test`: the CIT of Cai, Li and Zhang (2022), calibrated as
  above; its theory assumes a continuous scalar `Z`.
- `rp_ci.gcm.gcm_test`: the GCM of Shah and Peters (2020), with in-sample OLS
  residuals by default; pass `residualize` for another regression. It needs no
  reference pool, handles multivariate `Z`, and detects dependence only through
  the residual covariance of `X` and `Y` given `Z`. With OLS, its level requires
  `E[X | Z]` and `E[Y | Z]` to be linear in `Z`.

RP-GCM combines `rp_test` with `gcm_test`:

```python
from rp_ci.gcm import gcm_test

result = rp_test(X, Y, Z, lambda x, y, z: gcm_test(x, y, z).pvalue, k=100, seed=42)
```

`examples/cit_quickstart.py` and `examples/gcm_quickstart.py` run each test
alone and inside `rp_test`.

## Experiments

Scripts for the experiments in the paper are in `experiments/`; see
`experiments/README.md`.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```
