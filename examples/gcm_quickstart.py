"""Scalar GCM, RP-GCM, and top-r aggregation of the same projection p-values."""
import numpy as np

from rp_ci import aggregate, rp_test
from rp_ci.gcm import gcm_test


def main():
    rng = np.random.default_rng(42)
    n, k, r, alpha = 200, 1000, 10, .05
    z = rng.normal(size=(n, 5))
    x = z @ rng.normal(size=(5, 20)) + rng.normal(size=(n, 20))
    y = z @ rng.normal(size=(5, 20)) + rng.normal(size=(n, 20))

    scalar = gcm_test(x[:, 0], y[:, 0], z, alpha=alpha)
    print(f"Scalar GCM: p={scalar.pvalue:.3f}, reject={scalar.reject}")

    def base_test(x, y, z):
        return gcm_test(x, y, z).pvalue

    projected = rp_test(x, y, z, base_test, k=k, alpha=alpha, seed=202)
    print(f"RP-GCM, max: min p={projected.min_pvalue:.4f}, reject={projected.reject}")

    topr = aggregate(projected.pvalues, alpha=alpha, aggregation="beta", r=r)
    print(f"RP-GCM, top-{r}: r-th p={topr.rth_pvalue:.4f}, reject={topr.reject}")


if __name__ == "__main__":
    main()
