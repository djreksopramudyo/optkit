# Black-Scholes-Merton European option pricing.

# Closed-form pricing for European calls and puts under the
# Black-Scholes-Merton model with a continuous dividend yield.


import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    # Return the d1 and d2 terms shared by price and Greeks.
    # d1 = [ln(S/K) + (r - q + sigma**2 / 2) * T] / (sigma * sqrt(T))
    # d2 = d1 - sigma * sqrt(T)

    d1 = (np.log(S/K) + (r - q + sigma**2 / 2 * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(S, K, T, r, sigma, q=0.0, option="call"):
    # Price a European option under Black-Scholes-Merton.

    # Parameters
    # ----------
    # S : float       spot price of the underlying
    # K : float       strike price
    # T : float       time to expiry, in years
    # r : float       risk-free rate, continuously compounded (e.g. 0.05)
    # sigma : float   annualised volatility (e.g. 0.20)
    # q : float       continuous dividend yield (default 0.0)
    # option : str    "call" or "put"

    # Returns
    # -------
    # float : the option price

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option == "call":
        price = (S * np.e**(-q * T) * norm.cdf(d1)
                 - K * np.e**(-r * T) * norm.cdf(d2))
    elif option == "put":
        price = (K * np.e**(-r * T) * norm.cdf(-d2)
                 - S * np.e**(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")

    return price


def _pdf(x):
    # Standard normal PDF, n(x) = exp(-x**2 / 2) / sqrt(2*pi).
    return norm.pdf(x)


def bsm_delta(S, K, T, r, sigma, q=0.0, option="call"):
    # dPrice/dS.

    # Call: exp(-qT) * N(d1)
    # Put:  exp(-qT) * (N(d1) - 1)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return np.e**(-q * T) * norm.cdf(d1)
    elif option == "put":
        return np.e**(-q * T) * (norm.cdf(d1) - 1)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bsm_gamma(S, K, T, r, sigma, q=0.0):
    # d2Price/dS2 (same for calls and puts).

    # exp(-qT) * n(d1) / (S * sigma * sqrt(T))

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.e**(-q * T) * _pdf(d1) / ((S * sigma * np.sqrt(T)))


def bsm_vega(S, K, T, r, sigma, q=0.0):
    # dPrice/dSigma (same for calls and puts), raw (per 1.00 vol).

    # S * exp(-qT) * n(d1) * sqrt(T)

    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.e**(-q * T) * _pdf(d1) * np.sqrt(T)


def bsm_theta(S, K, T, r, sigma, q=0.0, option="call"):
    # dPrice/dT with the time-decay sign convention, raw (per year).

    # Shared term:
    #    term1 = -exp(-qT) * S * n(d1) * sigma / (2 * sqrt(T))
    # Call: term1 - r*K*exp(-rT)*N(d2) + q*S*exp(-qT)*N(d1)
    # Put:  term1 + r*K*exp(-rT)*N(-d2) - q*S*exp(-qT)*N(-d1)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -np.exp(-q * T) * S * _pdf(d1) * sigma / (2 * np.sqrt(T))
    if option == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
                + q * S * np.exp(-q * T) * norm.cdf(d1))
    elif option == "put":
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1))
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bsm_rho(S, K, T, r, sigma, q=0.0, option="call"):
    # dPrice/dr, raw (per 1.00 rate).

    # Call:  K * T * exp(-rT) * N(d2)
    # Put:  -K * T * exp(-rT) * N(-d2)

    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return K * T * np.e**(-r * T) * norm.cdf(d2)
    elif option == "put":
        return -K * T * np.e**(-r * T) * norm.cdf(-d2)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")
