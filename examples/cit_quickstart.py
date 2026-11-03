"""Scalar CIT and a shared-reference callback for the RP framework."""
import numpy as np

from rp_ci import rp_test
from rp_ci.cit import cit_test, simulate_null_pool


def main():
    rng = np.random.default_rng(42)
    n, k, alpha, rp_h_scale = 40, 5, .05, .8
    z = rng.normal(size=(n, 1))
    x = z + rng.normal(size=(n, 3))
    y = z + rng.normal(size=(n, 3))

    pool = simulate_null_pool(n, 1, 1, 1, B=999, seed=101, n_jobs=1)
    scalar = cit_test(x[:, 0], y[:, 0], z, null_pool=pool)
    print(f"Scalar CIT: p={scalar.pvalue:.3f}, reject={scalar.reject}")

    def base_test(x, y, z):
        return cit_test(
            x, y, z, null_pool=pool, alpha=alpha / k,
            h_scale=rp_h_scale,
        ).pvalue

    projected = rp_test(x, y, z, base_test, k=k, alpha=alpha, seed=202)
    print(f"RP + CIT: min p={projected.min_pvalue:.3f}, reject={projected.reject}")


if __name__ == "__main__":
    main()
