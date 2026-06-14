# Tests for the Black-Scholes-Merton pricer.

import numpy as np
import pytest

from optkit.black_scholes import bsm_price


def test_known_textbook_value():
    # Canonical sanity check: S=K=100, T=1, r=5%, sigma=20%, q=0.
    call = bsm_price(100, 100, 1, 0.05, 0.20, option="call")
    put = bsm_price(100, 100, 1, 0.05, 0.20, option="put")
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


@pytest.mark.parametrize(
    # S, K, T, r, sigma, q"
    [
        (100, 100, 1.0, 0.05, 0.20, 0.0),
        (120, 100, 0.5, 0.03, 0.35, 0.02),
        (90, 110, 2.0, 0.01, 0.15, 0.04),
        (100, 100, 0.25, 0.06, 0.50, 0.00),
    ]
)
def test_put_call_parity(S, K, T, r, sigma, q):
    # C - P must equal S*exp(-qT) - K*exp(-rT) for any inputs.
    call = bsm_price(S, K, T, r, sigma, q=q, option="call")
    put = bsm_price(S, K, T, r, sigma, q=q, option="put")
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_invalid_option_type():
    # Anything other than call/put should raise.
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0.20, option="banana")
