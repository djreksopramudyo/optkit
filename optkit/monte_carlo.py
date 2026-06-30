# Monte Carlo pricing for European options under risk-neutral GBM.
#
# Estimates  price = e^(-rT) * E[payoff(S_T)]  by averaging discounted payoffs
# over sampled terminal prices, with optional antithetic and control variates.

from dataclasses import dataclass
import numpy as np


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int  # independent observations; n_paths/2 when antithetic


def _simulate_terminal(S, T, r, sigma, q, Z):
    # Exact lognormal terminal price; -sigma²/2 is the Ito drift correction.
    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    return S_T


def _payoff(S_T, K, option):
    if option == "call":
        return np.maximum(S_T - K, 0)
    elif option == "put":
        return np.maximum(K - S_T, 0)
    raise ValueError(f"unknown option type: {option}")


def mc_price(S, K, T, r, sigma, q=0.0, option="call",
             n_paths=100_000, antithetic=True, control_variate=False,
             seed=None):
    # Monte Carlo price of a European option, with optional variance reduction.
    rng = np.random.default_rng(seed)
    half = n_paths // 2

    if antithetic:
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    S_T = _simulate_terminal(S, T, r, sigma, q, Z)
    pay = _payoff(S_T, K, option)

    # Control variate: correct each payoff by the terminal price's deviation
    # from its known mean. Must run per-path, before antithetic pairing.
    if control_variate:
        Y = S_T
        EY = S * np.exp((r - q) * T)
        beta = np.cov(pay, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)
        pay = pay - beta * (Y - EY)

    # Each antithetic pair is one i.i.d. observation, so the SE is computed
    # over the pair-averages, not the raw paths.
    if antithetic:
        obs = 0.5 * (pay[:half] + pay[half:])
    else:
        obs = pay

    disc = np.exp(-r * T)
    price = disc * obs.mean()
    se = disc * obs.std(ddof=1) / np.sqrt(len(obs))
    return MCResult(price, se, price - 1.96 * se, price + 1.96 * se, len(obs))
