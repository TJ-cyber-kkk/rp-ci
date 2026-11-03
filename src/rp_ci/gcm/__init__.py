"""GCM base test."""

from .api import GCMResult, gcm_test
from .regression import ols_residuals

__all__ = ["GCMResult", "gcm_test", "ols_residuals"]
