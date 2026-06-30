# Black-Scholes-Merton European option pricing with analytical Greeks.

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _pdf(x):
    return norm.pdf(x)


def bsm_price(S, K, T, r, sigma, q=0.0, option="call"):
    # Price a European option under BSM with continuous dividend yield q.
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return (S * np.exp(-q * T) * norm.cdf(d1)
                - K * np.exp(-r * T) * norm.cdf(d2))
    elif option == "put":
        return (K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1))
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bsm_delta(S, K, T, r, sigma, q=0.0, option="call"):
    # dPrice/dS.
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bsm_gamma(S, K, T, r, sigma, q=0.0):
    # d²Price/dS², identical for calls and puts.
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * _pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S, K, T, r, sigma, q=0.0):
    # dPrice/dSigma per 1.0 vol move, identical for calls and puts.
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * _pdf(d1) * np.sqrt(T)


def bsm_theta(S, K, T, r, sigma, q=0.0, option="call"):
    # Time decay per year (= -dPrice/dT). Usually negative for long options.
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    decay = -np.exp(-q * T) * S * _pdf(d1) * sigma / (2 * np.sqrt(T))
    if option == "call":
        return (decay
                - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1))
    elif option == "put":
        return (decay
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1))
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bsm_rho(S, K, T, r, sigma, q=0.0, option="call"):
    # dPrice/dr per 1.0 rate move.
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")
