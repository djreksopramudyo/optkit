# Tests for the implied volatility solver.
#
# Core check is a round-trip: price at a known sigma, recover it, assert they
# match. Edge cases cover the no-arb bounds and the Newton/bisection fallback.

import numpy as np
import pytest

from optkit.black_scholes import bsm_price
from optkit.implied_vol import implied_vol


# Spanning ITM / ATM / OTM, short/long dated, with and without dividends.
CASES = [
    (100, 100, 1.0, 0.05, 0.0, "call"),
    (100, 100, 1.0, 0.05, 0.0, "put"),
    (100, 90,  0.5, 0.03, 0.0, "call"),
    (100, 120, 0.5, 0.03, 0.0, "call"),
    (100, 120, 2.0, 0.01, 0.02, "put"),
    (100, 80,  0.25, 0.05, 0.0, "put"),
]
SIGMAS = [0.10, 0.20, 0.35, 0.60]


@pytest.mark.parametrize("S,K,T,r,q,option", CASES)
@pytest.mark.parametrize("true_sigma", SIGMAS)
def test_round_trip_auto(S, K, T, r, q, option, true_sigma):
    price = bsm_price(S, K, T, r, true_sigma, q=q, option=option)
    res = implied_vol(price, S, K, T, r, q=q, option=option, method="auto")
    assert res.converged
    assert res.iv == pytest.approx(true_sigma, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,q,option", CASES)
@pytest.mark.parametrize("true_sigma", SIGMAS)
def test_newton_and_bisection_agree(S, K, T, r, q, option, true_sigma):
    price = bsm_price(S, K, T, r, true_sigma, q=q, option=option)
    n = implied_vol(price, S, K, T, r, q=q, option=option, method="newton")
    b = implied_vol(price, S, K, T, r, q=q, option=option, method="bisection")
    assert n.iv == pytest.approx(b.iv, abs=1e-4)


def test_price_above_upper_bound_returns_nan():
    S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
    upper = S * np.exp(-q * T)
    res = implied_vol(upper + 1.0, S, K, T, r, q=q, option="call")
    assert np.isnan(res.iv) and not res.converged


def test_price_below_intrinsic_returns_nan():
    S, K, T, r, q = 100, 50, 1.0, 0.05, 0.0
    intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
    res = implied_vol(intrinsic - 1.0, S, K, T, r, q=q, option="call")
    assert np.isnan(res.iv)


def test_otm_recovers_via_auto():
    # OTM call: low vega, where Newton is fragile & the backup earns its keep.
    S, K, T, r, q = 100, 140, 0.5, 0.05, 0.0
    true_sigma = 0.45
    price = bsm_price(S, K, T, r, true_sigma, q=q, option="call")
    res = implied_vol(price, S, K, T, r, q=q, option="call", method="auto")
    assert res.converged
    assert res.iv == pytest.approx(true_sigma, abs=1e-3)
