# Implied volatility solver for European options under Black-Scholes-Merton.
#
# Inverse of pricing: given a market price, find sigma with bsm_price(sigma)
# == price. No closed form, so solve f(sigma) = bsm_price(sigma) - price = 0
# numerically. f is increasing in sigma (vega > 0), so the root is unique
# whenever the price sits inside the no-arbitrage bounds.
#
# Newton-Raphson runs first (fast, uses vega as f'); bisection is the fallback
# (cannot diverge) when Newton stalls near zero vega.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from optkit.black_scholes import bsm_price, bsm_vega


@dataclass
class IVResult:
    iv: float
    converged: bool
    method: str
    iterations: int


# Sigma search range for bisection. 1e-6 avoids the zero-vol degenerate case;
# 5.0 (500% vol) brackets any in-bounds price.
_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0


def _no_arb_bounds(S, K, T, r, q, option):
    # No-arbitrage price band. Inside it exactly one implied vol exists; at or
    # outside, none does and the caller returns NaN.
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option == "call":
        return max(disc_S - disc_K, 0.0), disc_S
    return max(disc_K - disc_S, 0.0), disc_K


def _newton(price, S, K, T, r, q, option, sigma0, tol, max_iter):
    # Newton-Raphson on f(sigma) = bsm_price(sigma) - price, with vega as f'.
    # Returns (sigma, converged, iters); converged=False tells the caller to
    # fall back rather than trust the value.
    sigma = sigma0
    for i in range(1, max_iter + 1):
        diff = bsm_price(S, K, T, r, sigma, q=q, option=option) - price
        if np.abs(diff) < tol:
            return (sigma, True, i)
        v = bsm_vega(S, K, T, r, sigma, q=q)
        if v < 1e-8:
            return (sigma, False, i)
        sigma -= diff / v
        if sigma <= 0 or sigma > _SIGMA_HI:
            return (sigma, False, i)
    return (sigma, False, max_iter)


def _bisection(price, S, K, T, r, q, option, tol, max_iter):
    # Bisection on the same f. [_SIGMA_LO, _SIGMA_HI] always brackets the root
    # for an in-bounds price, since f is monotonic in sigma.
    lo, hi = _SIGMA_LO, _SIGMA_HI
    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        diff = bsm_price(S, K, T, r, mid, q=q, option=option) - price
        if np.abs(diff) < tol or (hi - lo) < tol:
            return (mid, True, i)
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (0.5 * (lo + hi), False, max_iter)


def implied_vol(price, S, K, T, r, sigma0=0.2, q=0.0, option="call",
                method="auto", tol=1e-8, max_iter=100):
    # Solve for implied vol. method="auto" tries Newton then bisection;
    # "newton"/"bisection" force one solver. iv is NaN when the price is out
    # of bounds or nothing converged.
    lower, upper = _no_arb_bounds(S, K, T, r, q, option)
    if not (lower < price < upper):
        return IVResult(np.nan, False, "none", 0)

    if method in ("auto", "newton"):
        sigma, ok, it = _newton(price, S, K, T, r, q, option,
                                sigma0, tol, max_iter)
        if ok:
            return IVResult(sigma, True, "newton", it)
        if method == "newton":
            return IVResult(np.nan, False, "newton", it)

    sigma, ok, it = _bisection(price, S, K, T, r, q, option, tol, max_iter)
    return IVResult(sigma if ok else np.nan, ok, "bisection", it)