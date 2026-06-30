# Tests for the Monte Carlo pricer and its variance-reduction paths.

import numpy as np
import pytest

from optkit.black_scholes import bsm_price
from optkit.monte_carlo import mc_price

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_within_ci_of_bsm(option):
    # BSM price lands inside the MC 95% CI at large n_paths.
    bsm = bsm_price(S, K, T, r, sigma, q=q, option=option)
    res = mc_price(S, K, T, r, sigma, q=q, option=option,
                   n_paths=400_000, antithetic=False, seed=0)
    assert res.ci_low <= bsm <= res.ci_high


@pytest.mark.parametrize("option", ["call", "put"])
def test_se_shrinks_with_paths(option):
    # SE falls as n_paths grows: the reliable convergence signal (realized
    # error is noisy and not monotone, so it is not asserted directly).
    small = mc_price(S, K, T, r, sigma, q=q, option=option,
                     n_paths=10_000, antithetic=False, seed=1)
    large = mc_price(S, K, T, r, sigma, q=q, option=option,
                     n_paths=400_000, antithetic=False, seed=1)
    assert large.std_error < small.std_error


def test_antithetic_reduces_se():
    # Antithetic SE < plain SE at the same n_paths.
    plain = mc_price(S, K, T, r, sigma, q=q, option="call",
                     n_paths=100_000, antithetic=False, seed=2)
    anti = mc_price(S, K, T, r, sigma, q=q, option="call",
                    n_paths=100_000, antithetic=True, seed=2)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_se():
    # Control-variate SE < plain SE at the same n_paths.
    plain = mc_price(S, K, T, r, sigma, q=q, option="call",
                     n_paths=100_000, antithetic=False,
                     control_variate=False, seed=3)
    cv = mc_price(S, K, T, r, sigma, q=q, option="call",
                  n_paths=100_000, antithetic=False,
                  control_variate=True, seed=3)
    assert cv.std_error < plain.std_error
