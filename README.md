# cedardev-fixed-income

[![PyPI version](https://badge.fury.io/py/cedardev-fixed-income.svg)](https://badge.fury.io/py/cedardev-fixed-income)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Industrial-grade fixed income risk analytics for quantitative researchers and risk infrastructure teams.**

Built by **CedarDev Capital Management LLC** — Market Risk Infrastructure desk.

---

## What's in the box

| Module | What it does |
|---|---|
| `ust_pricer` | US Treasury pricing: clean/dirty price, YTM, duration, DV01, convexity, zero-curve bootstrap, Z-spread |
| `mbs_analytics` | Agency MBS: PSA prepayment model, cash flow generation, WAL, OAS, PSA sensitivity |
| `treasury_futures` | CME Treasury futures: CTD identification, conversion factor, implied repo, basis |
| `var_engine` | VaR: Historical Simulation (delta-gamma), Parametric (EWMA), Expected Shortfall, Component VaR |
| `margin_engine` | SEC Rule 4210 margining pipeline: extract → categorize → net → cross-margin → validate |
| `model_validation` | Backtesting: Kupiec POF, Christoffersen independence, Basel traffic light, FRTB PLA test |

---

## Installation

```bash
pip install cedardev-fixed-income
```

---

## Quick Start

### Price a US Treasury bond

```python
from datetime import date
from cedardev.fixed_income import USTBond, USTBondPricer, ZeroCurve

# Define the bond
bond = USTBond(
    cusip         = "91282CEX5",
    issue_date    = date(2021, 8, 15),
    maturity_date = date(2031, 7, 31),
    coupon_rate   = 0.0125,   # 1.25%
)

# Build a zero curve (from on-the-run par yields)
par_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
par_yields = [0.0530, 0.0528, 0.0520, 0.0490, 0.0475, 0.0450,
              0.0440, 0.0430, 0.0445, 0.0435]
zero_curve = ZeroCurve.bootstrap(par_tenors, par_yields)

# Price and get risk measures
pricer     = USTBondPricer()
settlement = date(2024, 7, 1)
analytics  = pricer.full_analytics(bond, clean_price_100=84.10, settlement=settlement,
                                   zero_curve=zero_curve)

print(f"YTM             : {analytics.ytm:.4%}")
print(f"Modified Duration: {analytics.modified_duration:.4f} years")
print(f"DV01            : ${analytics.dv01 * 1_000_000:,.0f} per $1MM face per bp")
print(f"Z-spread        : {analytics.z_spread:.1f} bps")
```

### Price an Agency MBS pool

```python
from datetime import date
from cedardev.fixed_income import MBSPool, MBSPricer, AgencyType, CollateralType
import numpy as np

# Zero curve
zc_tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
zc_rates  = np.array([0.0528, 0.0526, 0.0518, 0.0488, 0.0472, 0.0447,
                      0.0437, 0.0427, 0.0432, 0.0440, 0.0437, 0.0433])

pool = MBSPool(
    pool_number      = "MA3456",
    agency           = AgencyType.FNMA,
    collateral_type  = CollateralType.FIXED_RATE,
    original_balance = 50_000_000,
    current_factor   = 0.85,
    gross_coupon     = 0.065,
    net_coupon       = 0.0625,
    wam              = 320,
    wala             = 40,
    issue_date       = date(2021, 3, 1),
    settlement_date  = date(2024, 7, 1),
)

pricer    = MBSPricer(zc_tenors, zc_rates)
analytics = pricer.full_analytics(pool, price_100=101.25, psa_speed=150.0)

print(f"OAS    : {analytics.oas_bps:.1f} bps")
print(f"WAL    : {analytics.wal_years:.2f} years")
print(f"OA Dur : {analytics.modified_duration:.3f}")
print(f"DV01   : ${analytics.dv01 * pool.current_balance:,.0f}")
```

### Run the SEC Rule 4210 Margin Engine

```python
from cedardev.fixed_income import MarginPipeline
from pathlib import Path

pipeline = MarginPipeline(
    use_synthetic = True,       # swap for False in production (uses Snowflake)
    output_dir    = Path("./margin_output"),
)
results_df, validation = pipeline.run()

# results_df columns:
# portfolio_id | bucket | gross_margin | net_margin | cross_margin_credit | final_margin
```

### Backtest a VaR model

```python
from cedardev.fixed_income import (
    ModelValidator, BacktestWindow, KupiecTest, BaselTrafficLightAssigner
)
import numpy as np
from datetime import date, timedelta

# Simulated backtest data (replace with real desk P&L)
np.random.seed(42)
n          = 500
var_series = np.full(n, 500_000 * 2.326)   # 99% VaR estimate
actual_pnl = np.random.normal(0, 500_000, n)

backtest = [
    BacktestWindow(
        date         = date(2023, 1, 1) + timedelta(days=i),
        var_estimate = var_series[i],
        actual_pnl   = actual_pnl[i],
    )
    for i in range(n)
]

validator = ModelValidator()
report    = validator.run(
    backtest_data    = backtest,
    hypothetical_pnl = actual_pnl * 0.95,
    actual_pnl       = actual_pnl,
)
validator.print_report(report)
# → Kupiec, Christoffersen, Basel traffic light, PLA test results
```

---

## Design principles

- **No black boxes.** Every calculation is traceable to its formula. Comments reference Fabozzi, Basel documents, and SEC releases.
- **No ML.** Traditional quantitative methods only — Pandas, NumPy, SciPy. Exactly what regulators expect.
- **Audit trails.** Every `MarginResult` carries `audit_notes` with the full dollar breakdown.
- **Regulatory-first.** Day-count conventions, bucket boundaries, margin rates, and correlation matrices all sourced from published SEC/FINRA/CME documentation.
- **Snowflake-native.** The margin engine queries `RATES_DW.POSITIONS.OPEN_POSITIONS_V` by default, with a `--synthetic` flag for testing without a live DB connection.

---

## Regulatory coverage

| Standard | Where implemented |
|---|---|
| SEC Rule 4210 (FINRA) | `margin_engine.py` |
| Basel II.5 / III internal models | `var_engine.py` |
| Basel backtesting framework (1996) | `model_validation.py` |
| FRTB P&L Attribution test | `model_validation.py` |
| CME Treasury futures delivery specs | `treasury_futures.py` |
| SIFMA PSA prepayment standard | `mbs_analytics.py` |
| ACT/ACT ICMA (ISMA Rule 251) | `ust_pricer.py` |

---

## Requirements

- Python 3.11+
- pandas, numpy, scipy, pydantic, tabulate, colorlog
- snowflake-connector-python (only needed for live DB extraction)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Disclaimer

This library is intended for internal quantitative research and risk infrastructure use.
All regulatory table values are illustrative and based on publicly available SEC/FINRA documentation.
Always verify against your firm's current regulatory schedule before using in production.
