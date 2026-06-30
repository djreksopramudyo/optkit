# Tests for the Black-Scholes-Merton pricer.

import numpy as np
import pytest
from optkit.black_scholes import (
    bsm_price, bsm_delta, bsm_gamma, bsm_vega, bsm_theta, bsm_rho,
)


def test_known_textbook_value():
    call = bsm_price(100, 100, 1, 0.05, 0.20, option="call")
    put = bsm_price(100, 100, 1, 0.05, 0.20, option="put")
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


@pytest.mark.parametrize(
    "S, K, T, r, sigma, q",
    [
        (100, 100, 1.0, 0.05, 0.20, 0.0),
        (120, 100, 0.5, 0.03, 0.35, 0.02),
        (90, 110, 2.0, 0.01, 0.15, 0.04),
        (100, 100, 0.25, 0.06, 0.50, 0.00),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bsm_price(S, K, T, r, sigma, q=q, option="call")
    put = bsm_price(S, K, T, r, sigma, q=q, option="put")
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0.20, option="banana")


BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)


def _price(option=None, **kw):
    p = {**BASE, **kw}
    return bsm_price(p["S"], p["K"], p["T"], p["r"], p["sigma"],
                     q=p["q"], option=option)


@pytest.mark.parametrize("option", ["call", "put"])
def test_delta_matches_finite_diff(option):
    h = 1e-4
    fd = ((_price(option, S=BASE["S"] + h)
           - _price(option, S=BASE["S"] - h)) / (2 * h))
    assert bsm_delta(**BASE, option=option) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("option", ["call", "put"])
def test_gamma_matches_finite_diff(option):
    h = 1e-2
    second = (_price(option, S=BASE["S"] + h)
              - 2 * _price(option, S=BASE["S"])
              + _price(option, S=BASE["S"] - h)) / h**2
    assert bsm_gamma(**BASE) == pytest.approx(second, abs=1e-5)


@pytest.mark.parametrize("option", ["call", "put"])
def test_vega_matches_finite_diff(option):
    h = 1e-4
    fd = (_price(option, sigma=BASE["sigma"] + h)
          - _price(option, sigma=BASE["sigma"] - h)) / (2 * h)
    assert bsm_vega(**BASE) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option", ["call", "put"])
def test_theta_matches_finite_diff(option):
    h = 1e-4
    fd = (-(_price(option, T=BASE["T"] + h)
            - _price(option, T=BASE["T"] - h)) / (2 * h))
    assert bsm_theta(**BASE, option=option) == pytest.approx(fd, abs=1e-4)


@pytest.mark.parametrize("option", ["call", "put"])
def test_rho_matches_finite_diff(option):
    h = 1e-4
    fd = ((_price(option, r=BASE["r"] + h)
           - _price(option, r=BASE["r"] - h)) / (2 * h))
    assert bsm_rho(**BASE, option=option) == pytest.approx(fd, abs=1e-4)
