"""CIT base test, its numerical components, and plus-one calibration."""

from .api import CITResult, cit_test
from .calibration import plus_one_critical_value, plus_one_pvalues, simulate_null_pool
from .direct_fast import direct_cit_statistic_fast, rosenblatt_target_given_z_fast
from .rosenblatt import band_width, rosenblatt_transform
from .scalar_batch import scalar_projected_statistics
from .statistic import RhoHatCachedW, rho_hat_multivariate

__all__ = [
    "CITResult", "cit_test", "band_width", "rosenblatt_transform",
    "direct_cit_statistic_fast", "rosenblatt_target_given_z_fast",
    "rho_hat_multivariate", "RhoHatCachedW", "scalar_projected_statistics",
    "simulate_null_pool", "plus_one_pvalues", "plus_one_critical_value",
]
