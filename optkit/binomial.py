# Cox-Ross-Rubinstein binomial tree pricing for European and American options.

# Recombining lattice: at step i there are i+1 distinct nodes (not 2^i),
# because up-then-down and down-then-up land on the same price.
# The backward induction runs in O(N²) time and O(N) memory.


import numpy as np


def _crr_params(T, r, sigma, N, q=0.0):
    # Return (dt, u, d, p, disc) for the CRR parameterization.

    # u/d are chosen so the tree variance matches BSM as dt→0.
    # p is the risk-neutral probability, not a real-world forecast.

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    return dt, u, d, p, disc


def binomial_price(S, K, T, r, sigma, N, q=0.0, option="call", american=False):
    # Price a European or American option via the CRR binomial lattice.

    # Parameters
    # ----------
    # S, K    : spot and strike price
    # T       : time to expiry in years
    # r, q    : risk-free rate and continuous dividend yield
    # sigma   : annualised volatility
    # N       : number of time steps
    # option  : "call" or "put"
    # american: if True, checks early exercise at every node

    dt, u, d, p, disc = _crr_params(T, r, sigma, N, q)

    # Terminal stock prices and payoffs at expiry
    j = np.arange(N + 1)
    S_T = S * u**j * d**(N - j)

    if option == "call":
        V = np.maximum(S_T - K, 0)
    elif option == "put":
        V = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option must be 'call' or 'put'")

    # Backward induction: V[1:] and V[:-1] are the up/down children
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])

        if american:
            j_i = np.arange(i + 1)
            S_i = S * u**j_i * d**(i - j_i)
            intrinsic = (np.maximum(S_i - K, 0) if option == "call"
                         else np.maximum(K - S_i, 0))
            V = np.maximum(V, intrinsic)

    return V[0]
