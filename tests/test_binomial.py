# Tests for the CRR binomial pricer.

import pytest
import numpy as np
from optkit.binomial import binomial_price
from optkit.black_scholes import bsm_price


S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
N_CHECK = 1000
TOL = 0.01


def test_european_call_matches_bsm():
    crr = binomial_price(S, K, T, r, sigma, N=N_CHECK, option="call")
    bsm = bsm_price(S, K, T, r, sigma, option="call")
    assert abs(crr - bsm) < TOL


def test_european_put_matches_bsm():
    crr = binomial_price(S, K, T, r, sigma, N=N_CHECK, option="put")
    bsm = bsm_price(S, K, T, r, sigma, option="put")
    assert abs(crr - bsm) < TOL


def test_american_put_ge_european_put():
    eur = binomial_price(S, K, T, r, sigma, N=N_CHECK,
                         option="put", american=False)
    amer = binomial_price(S, K, T, r, sigma, N=N_CHECK,
                          option="put", american=True)
    assert amer >= eur - 1e-8


def test_american_call_equals_european_call_no_dividend():
    eur = binomial_price(S, K, T, r, sigma, N=N_CHECK,
                         option="call", american=False)
    amer = binomial_price(S, K, T, r, sigma, N=N_CHECK,
                          option="call", american=True)
    assert abs(amer - eur) < TOL


def test_convergence():
    bsm = bsm_price(S, K, T, r, sigma, option="call")
    errors = [
        abs(binomial_price(S, K, T, r, sigma, N=n, option="call") - bsm)
        for n in [10, 50, 200, 1000]
    ]
    assert all(errors[i] > errors[i+1] for i in range(len(errors)-1))
