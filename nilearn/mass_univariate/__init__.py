"""
Defines a Massively Univariate Linear Model estimated with OLS and permutation test
"""

from .permuted_least_squares import permuted_ols
from .rpbi import randomized_parcellation_based_inference

__all__ = ['permuted_ols', 'randomized_parcellation_based_inference']
