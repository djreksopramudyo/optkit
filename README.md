# optkit

A from-scratch options pricing and Greeks toolkit in Python.

## Overview

`optkit` implements European option pricing under the Black-Scholes-Merton
model, including a continuous dividend yield, along with the standard
first- and second-order Greeks (delta, gamma, vega, theta, rho). All
analytical Greeks are validated against finite-difference approximations.

## Installation

```bash
git clone https://github.com/djreksopramudyo/optkit.git
cd optkit
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Usage

```python
from optkit.black_scholes import bsm_price, bsm_delta

bsm_price(100, 100, 1, 0.05, 0.20, option="call")  # ~10.45
bsm_delta(100, 100, 1, 0.05, 0.20, option="call")
```

## Testing

```bash
pytest -v
```

## Roadmap

- [x] BSM European pricer + analytical Greeks (finite-difference validated)
- [x] Binomial tree (European + American)
- [ ] Monte Carlo with variance reduction
- [ ] Implied volatility solver
- [ ] Volatility smile & surface plots
