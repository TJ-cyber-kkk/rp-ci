import numpy as np

from rp_ci import rp_test


def placeholder_ci_test(x, y, z):
    return 1.0


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    Y = rng.normal(size=(100, 3))
    Z = rng.normal(size=(100, 1))

    result = rp_test(
        X, Y, Z,
        base_test=placeholder_ci_test,
        k=10,
        alpha=0.05,
        seed=42,
    )

    print("Interface demo only: the placeholder always returns p = 1.0.")
    print("Replace it with a calibrated scalar CI test for statistical analysis.")
    print(f"Reject: {result.reject}")
    print(f"Minimum p-value (unadjusted): {result.min_pvalue}")
    print(f"Bonferroni threshold: {result.threshold}")
    print(f"Projection p-values: {result.pvalues}")


if __name__ == "__main__":
    main()
