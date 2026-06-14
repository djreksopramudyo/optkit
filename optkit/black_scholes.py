"""Black-Scholes-Merton European option pricing.

Closed-form pricing for European calls and puts under the
Black-Scholes-Merton model with a continuous dividend yield.
"""

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
    """Price a European option under Black-Scholes-Merton.

    Parameters
    ----------
    S : float       spot price of the underlying
    K : float       strike price
    T : float       time to expiry, in years
    r : float       risk-free rate, continuously compounded (e.g. 0.05)
    sigma : float   annualised volatility (e.g. 0.20)
    q : float       continuous dividend yield (default 0.0)
    option : str    "call" or "put"

    Returns
    -------
    float : the option price
    """
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
