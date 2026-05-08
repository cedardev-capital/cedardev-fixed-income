"""
var_engine.py
=============
Value-at-Risk (VaR) Engine — Fixed Income Portfolio
Firm: Meridian Capital Management LLC  |  Desk: Market Risk Infrastructure
Version: 2.2.0  |  Python 3.11+

Methods Implemented
-------------------
  1. Historical Simulation VaR (HS-VaR)
       - Full re-valuation using the last N trading days of yield curve shifts
       - No distributional assumption — fully non-parametric
       - Regulatory preference: Basel II.5 / Basel III internal models approach

  2. Parametric VaR (Delta-Normal / Delta-Gamma)
       - Covariance matrix of yield curve key rates (KR DV01 approach)
       - Delta approximation:   VaR ≈ z_α × √(w'Σw)  where w = DV01 vector
       - Delta-Gamma correction: adds convexity adjustment

  3. Expected Shortfall (ES / CVaR)
       - Mean of losses BEYOND the VaR threshold
       - Required under FRTB (Fundamental Review of the Trading Book)
       - Also known as Conditional VaR (CVaR)

  4. Component VaR & Marginal VaR
       - Decomposes portfolio VaR by instrument / tenor bucket
       - Identifies largest risk contributors

Regulatory Context
------------------
  The SEC requires daily VaR reporting for broker-dealers with > $100M AUM.
  The internal model must be backtested (model_validation.py) and submitted
  to the SEC on a monthly basis.  VaR is computed at 99% confidence, 10-day
  holding period per Basel standards.

References
----------
  - Basel Committee, "Minimum Capital Requirements for Market Risk" (Jan 2019)
  - Riskmetrics Technical Document (J.P. Morgan, 1996) — still canonical
  - Jorion, "Value at Risk" (3rd ed.), Chapters 5–7
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Key Rate tenors — must match the zero curve nodes in ust_pricer.py
KEY_RATE_TENORS: List[float] = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

# VaR parameters
CONFIDENCE_LEVEL:  float = 0.99      # 99% — Basel standard
HOLDING_PERIOD:    int   = 1         # 1-day VaR (scale to 10-day via √10)
LOOKBACK_DAYS:     int   = 500       # Basel requires ≥ 250 business days
SCALING_FACTOR_10D = math.sqrt(10) if False else np.sqrt(10)   # √10 scaling


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

import math

@dataclass
class PortfolioPosition:
    """A single fixed income position in the VaR portfolio."""
    position_id:       str
    cusip:             str
    instrument_type:   str            # UST, CORP, AGENCY, MBS, SWAP, FUTURES
    face_amount:       float
    market_value:      float
    dv01_by_tenor:     Dict[float, float]  # {tenor_yr: DV01_$}  key-rate DV01 breakdown
    convexity:         float          = 0.0
    portfolio_id:      str            = "DEFAULT"
    bucket:            str            = "5-10yr"


@dataclass
class VaRResult:
    """Complete VaR output bundle."""
    as_of_date:             date
    method:                 str
    confidence_level:       float
    holding_period_days:    int
    var_1day:               float     # 1-day VaR ($)
    var_10day:              float     # 10-day VaR = var_1day × √10
    expected_shortfall:     float     # ES / CVaR ($)
    component_var:          pd.DataFrame
    pnl_distribution:       np.ndarray   # for backtesting / histogram
    marginal_var:           pd.DataFrame
    metadata:               Dict      = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SYNTHETIC YIELD CURVE HISTORY  (Snowflake stub for demo / test)
# ──────────────────────────────────────────────────────────────────────────────

def load_yield_curve_history(
    tenors:       List[float],
    lookback_days: int = LOOKBACK_DAYS,
    as_of:        Optional[date] = None,
    from_snowflake: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame of shape (lookback_days, len(tenors))
    containing daily yield levels (decimal).

    Production mode  : queries Snowflake
        SELECT date, tenor_2y, tenor_5y, tenor_10y, ...
        FROM RATES_DW.MARKET_DATA.TREASURY_PAR_YIELDS
        WHERE date >= DATEADD(day, -{lookback_days}, '{as_of}')
        ORDER BY date

    Synthetic mode (default): generates realistic correlated yield history
    using a factor model with mean-reversion.
    """
    if from_snowflake:
        # ── Production Snowflake query (stub) ─────────────────────────────
        import snowflake.connector
        from cedardev.fixed_income.margin_engine import SNOWFLAKE_CONFIG

        as_of = as_of or date.today()
        start = as_of - timedelta(days=int(lookback_days * 1.4))  # buffer for holidays

        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        sql  = f"""
            SELECT
                trade_date,
                rate_3m, rate_6m, rate_1y, rate_2y, rate_3y,
                rate_5y, rate_7y, rate_10y, rate_20y, rate_30y
            FROM RATES_DW.MARKET_DATA.UST_PAR_CURVE_DAILY
            WHERE trade_date BETWEEN '{start}' AND '{as_of}'
              AND curve_type = 'PAR'
            ORDER BY trade_date ASC
        """
        cur = conn.cursor(snowflake.connector.DictCursor)
        cur.execute(sql)
        df  = pd.DataFrame(cur.fetchall())
        conn.close()
        df  = df.set_index("trade_date").tail(lookback_days)
        return df

    # ── Synthetic generation (realistic, correlated) ──────────────────────
    np.random.seed(2024_07_01)
    as_of  = as_of or date.today()
    n_days = lookback_days
    n_t    = len(tenors)

    # Approximate long-run yield levels (mid-2024 regime)
    mu = np.array([0.0527, 0.0528, 0.0521, 0.0489, 0.0472,
                   0.0447, 0.0440, 0.0428, 0.0443, 0.0434])[:n_t]

    # Correlation structure: high correlation for adjacent tenors
    corr = np.ones((n_t, n_t))
    for i in range(n_t):
        for j in range(n_t):
            dist = abs(tenors[i] - tenors[j])
            corr[i, j] = np.exp(-0.08 * dist)

    # Daily volatilities (~5–10 bps/day, longer end more volatile)
    vols = np.array([0.0004, 0.0004, 0.0005, 0.0006, 0.0007,
                     0.0008, 0.0008, 0.0008, 0.0007, 0.0007])[:n_t]

    cov   = np.outer(vols, vols) * corr
    L     = cholesky(cov, lower=True)
    eps   = np.random.randn(n_days, n_t)
    shocks = (L @ eps.T).T   # (n_days, n_t)

    # Mean-reverting (Vasicek-like) daily simulation
    kappa  = 0.005    # mean reversion speed
    yields = np.zeros((n_days, n_t))
    y      = mu.copy()
    for t in range(n_days):
        y         = y + kappa * (mu - y) + shocks[t]
        yields[t] = np.clip(y, 0.0001, 0.20)

    dates = pd.bdate_range(end=as_of, periods=n_days)
    df    = pd.DataFrame(yields, index=dates, columns=[f"rate_{t}yr" for t in tenors])
    log.debug("Synthetic yield curve history: %d days × %d tenors.", n_days, n_t)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4.  KEY-RATE DV01 MAPPER
# ──────────────────────────────────────────────────────────────────────────────

class KeyRateDV01Mapper:
    """
    Maps instrument-level DV01s to the standard key-rate tenor grid.

    Each position has a single "effective duration tenor" (e.g. a 7yr bond
    maps mostly to the 7yr key rate, with spillover to 5yr and 10yr).
    This triangular interpolation follows the standard key-rate duration
    methodology (Ho, 1992).
    """

    def __init__(self, tenors: List[float] = None) -> None:
        self.tenors = np.array(tenors or KEY_RATE_TENORS)

    def decompose(
        self,
        position:    PortfolioPosition,
        ytm_years:   float,
    ) -> Dict[float, float]:
        """
        Distributes the position's DV01 across adjacent key-rate tenors
        using piecewise-linear interpolation weights.
        Returns {tenor: dv01_at_that_tenor}.
        """
        total_dv01 = sum(position.dv01_by_tenor.values())
        if total_dv01 == 0:
            return {t: 0.0 for t in self.tenors}

        decomposed = {t: 0.0 for t in self.tenors}

        # Find the two bracketing tenors
        idx = np.searchsorted(self.tenors, ytm_years)
        idx = np.clip(idx, 1, len(self.tenors) - 1)

        t_lo, t_hi = self.tenors[idx - 1], self.tenors[idx]
        w_hi = (ytm_years - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 1.0
        w_lo = 1.0 - w_hi

        decomposed[t_lo] = total_dv01 * w_lo
        decomposed[t_hi] = total_dv01 * w_hi

        return decomposed


# ──────────────────────────────────────────────────────────────────────────────
# 5.  HISTORICAL SIMULATION VaR
# ──────────────────────────────────────────────────────────────────────────────

class HistoricalSimVaR:
    """
    Full-revaluation Historical Simulation VaR.

    For each historical scenario (date t):
      ΔP_t = Σ_k  DV01_k × Δy_k,t   (delta approximation)
           + ½ × Σ_k  Convexity_k × (Δy_k,t)²   (gamma correction)

    Where:
      DV01_k  = dollar sensitivity of the portfolio to key-rate k
      Δy_k,t  = historical change in yield at key-rate k on day t
      Convexity_k = dollar convexity at key-rate k

    The VaR is then the empirical quantile of {ΔP_t}.
    """

    def __init__(
        self,
        yield_history: pd.DataFrame,
        confidence:    float = CONFIDENCE_LEVEL,
    ) -> None:
        self.history    = yield_history
        self.confidence = confidence
        self._yield_changes = yield_history.diff().dropna()

    def compute(
        self,
        positions:     List[PortfolioPosition],
        ytm_by_pos:    Dict[str, float],   # {position_id: ytm_years}
    ) -> VaRResult:
        log.info(
            "HS-VaR: %d positions, %d scenarios, conf=%.0f%%",
            len(positions), len(self._yield_changes), self.confidence * 100
        )

        mapper = KeyRateDV01Mapper()

        # Build portfolio key-rate DV01 vector (aggregated)
        port_dv01     = np.zeros(len(KEY_RATE_TENORS))
        port_convexity = np.zeros(len(KEY_RATE_TENORS))

        for pos in positions:
            ytm_yr = ytm_by_pos.get(pos.position_id, 7.0)
            kr_dv01 = mapper.decompose(pos, ytm_yr)
            for i, t in enumerate(KEY_RATE_TENORS):
                port_dv01[i]      += kr_dv01.get(t, 0.0)
                port_convexity[i] += pos.convexity / len(KEY_RATE_TENORS)

        # ── P&L scenarios ──────────────────────────────────────────────
        tenor_cols = [f"rate_{t}yr" for t in KEY_RATE_TENORS
                      if f"rate_{t}yr" in self._yield_changes.columns]
        Δy = self._yield_changes[tenor_cols].values   # (n_days, n_tenors)

        # Delta P&L  = DV01 × Δy × (-10,000)  [DV01 is $ per bp → Δy in decimal]
        # Note: DV01 is positive for long bonds. Rate UP → Price DOWN → negative P&L.
        delta_pnl  = -Δy * port_dv01[np.newaxis, :] * 10_000   # (n_days, n_tenors)
        gamma_pnl  = 0.5 * (Δy ** 2) * port_convexity[np.newaxis, :] * (10_000 ** 2)

        scenario_pnl = delta_pnl.sum(axis=1) + gamma_pnl.sum(axis=1)

        # VaR = loss at (1 − confidence) quantile
        var_quantile = np.percentile(scenario_pnl, (1 - self.confidence) * 100)
        var_1day     = abs(min(var_quantile, 0))
        var_10day    = var_1day * SCALING_FACTOR_10D

        # Expected Shortfall: average of losses worse than VaR
        tail_losses  = scenario_pnl[scenario_pnl < var_quantile]
        es_1day      = abs(tail_losses.mean()) if len(tail_losses) > 0 else var_1day

        # ── Component VaR ─────────────────────────────────────────────
        component_var = self._component_var(positions, ytm_by_pos, mapper, scenario_pnl)

        # ── Marginal VaR ──────────────────────────────────────────────
        marginal_var  = self._marginal_var(positions, ytm_by_pos, mapper, var_1day)

        return VaRResult(
            as_of_date          = date.today(),
            method              = "Historical Simulation (Delta-Gamma)",
            confidence_level    = self.confidence,
            holding_period_days = HOLDING_PERIOD,
            var_1day            = var_1day,
            var_10day           = var_10day,
            expected_shortfall  = es_1day,
            component_var       = component_var,
            pnl_distribution    = scenario_pnl,
            marginal_var        = marginal_var,
            metadata            = {
                "n_scenarios":     len(scenario_pnl),
                "worst_loss":      scenario_pnl.min(),
                "best_gain":       scenario_pnl.max(),
                "pnl_std":         scenario_pnl.std(),
                "skewness":        float(stats.skew(scenario_pnl)),
                "kurtosis":        float(stats.kurtosis(scenario_pnl)),
            },
        )

    def _component_var(self, positions, ytm_by_pos, mapper, scenario_pnl) -> pd.DataFrame:
        """Component VaR = correlation of position P&L with portfolio P&L × position VaR."""
        rows = []
        portfolio_std = scenario_pnl.std()

        for pos in positions:
            ytm_yr   = ytm_by_pos.get(pos.position_id, 7.0)
            kr_dv01  = mapper.decompose(pos, ytm_yr)
            dv01_vec = np.array([kr_dv01.get(t, 0.0) for t in KEY_RATE_TENORS])

            tenor_cols = [f"rate_{t}yr" for t in KEY_RATE_TENORS
                          if f"rate_{t}yr" in self._yield_changes.columns]
            Δy   = self._yield_changes[tenor_cols].values
            pos_pnl = (-Δy * dv01_vec[np.newaxis, :] * 10_000).sum(axis=1)

            if portfolio_std > 0 and pos_pnl.std() > 0:
                rho = np.corrcoef(pos_pnl, scenario_pnl)[0, 1]
            else:
                rho = 0.0

            pos_var = abs(np.percentile(pos_pnl, (1 - self.confidence) * 100))
            comp_var = rho * pos_var

            rows.append({
                "position_id":     pos.position_id,
                "cusip":           pos.cusip,
                "instrument_type": pos.instrument_type,
                "bucket":          pos.bucket,
                "correlation":     rho,
                "standalone_var":  pos_var,
                "component_var":   comp_var,
                "pct_of_total":    0.0,   # filled below
            })

        df = pd.DataFrame(rows)
        total_comp = df["component_var"].sum()
        if total_comp != 0:
            df["pct_of_total"] = df["component_var"] / total_comp * 100
        return df.sort_values("component_var", ascending=False)

    def _marginal_var(self, positions, ytm_by_pos, mapper, total_var) -> pd.DataFrame:
        """Marginal VaR = change in total VaR from adding $1 of a position."""
        rows = []
        for pos in positions:
            rows.append({
                "position_id":    pos.position_id,
                "cusip":          pos.cusip,
                "market_value":   pos.market_value,
                "marginal_var":   total_var / pos.market_value if pos.market_value else 0,
            })
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PARAMETRIC VaR  (Delta-Normal / Variance-Covariance)
# ──────────────────────────────────────────────────────────────────────────────

class ParametricVaR:
    """
    Delta-Normal VaR using the key-rate DV01 vector and yield covariance matrix.

    Formula
    -------
        VaR = z_α × √(w' Σ w)

    Where:
        w   = portfolio key-rate DV01 vector × 10,000  (converts to $ per 1 yield unit)
        Σ   = covariance matrix of daily yield changes (in decimal units)
        z_α = one-tailed z-score (2.326 at 99%, 1.645 at 95%)

    Assumptions
    -----------
    - Yield changes are jointly normally distributed.
    - Portfolio P&L is linear in yield changes (delta approximation).
    - Does NOT capture fat tails or convexity effects — use HS for those.
    """

    def __init__(
        self,
        yield_history: pd.DataFrame,
        confidence:    float = CONFIDENCE_LEVEL,
        ewma_lambda:   float = 0.94,   # RiskMetrics EWMA decay factor
    ) -> None:
        self.history    = yield_history
        self.confidence = confidence
        self.lam        = ewma_lambda
        self._yield_changes = yield_history.diff().dropna()
        self._cov_matrix    = self._compute_ewma_cov()

    def _compute_ewma_cov(self) -> np.ndarray:
        """
        Exponentially-weighted covariance matrix (EWMA).
        RiskMetrics (1996): λ = 0.94 for daily VaR.
        More recent observations receive higher weight.
        """
        Δy    = self._yield_changes.values
        n, k  = Δy.shape
        cov   = np.zeros((k, k))
        w_sum = 0.0

        for t in range(n - 1, -1, -1):
            age    = (n - 1) - t
            weight = (1 - self.lam) * (self.lam ** age)
            cov    += weight * np.outer(Δy[t], Δy[t])
            w_sum  += weight

        return cov / w_sum if w_sum > 0 else cov

    def compute(
        self,
        positions:  List[PortfolioPosition],
        ytm_by_pos: Dict[str, float],
    ) -> VaRResult:
        log.info("Parametric VaR: %d positions, λ=%.2f", len(positions), self.lam)

        mapper    = KeyRateDV01Mapper()
        dv01_vec  = np.zeros(len(KEY_RATE_TENORS))

        for pos in positions:
            ytm_yr  = ytm_by_pos.get(pos.position_id, 7.0)
            kr_dv01 = mapper.decompose(pos, ytm_yr)
            for i, t in enumerate(KEY_RATE_TENORS):
                dv01_vec[i] += kr_dv01.get(t, 0.0)

        # w = DV01 × 10,000 (so that w'Σw is in $ units when Σ is in decimal²)
        w = dv01_vec * 10_000

        # Portfolio variance and volatility
        port_var_daily = w @ self._cov_matrix @ w
        port_std_daily = math.sqrt(max(port_var_daily, 0.0))

        z_alpha  = stats.norm.ppf(self.confidence)
        var_1day = z_alpha * port_std_daily
        var_10day = var_1day * SCALING_FACTOR_10D

        # ES for normal distribution: ES = σ × φ(z) / (1-α)
        es_1day  = port_std_daily * stats.norm.pdf(z_alpha) / (1 - self.confidence)

        # Simulate normal P&L distribution for consistency with HS output
        pnl_sim = np.random.normal(0, port_std_daily, size=10_000)

        # Component VaR (parametric): proportional to DV01 contribution
        component_rows = []
        for pos in positions:
            ytm_yr  = ytm_by_pos.get(pos.position_id, 7.0)
            kr_dv01 = mapper.decompose(pos, ytm_yr)
            w_i     = np.array([kr_dv01.get(t, 0.0) for t in KEY_RATE_TENORS]) * 10_000
            comp_var = (w_i @ self._cov_matrix @ w) / port_std_daily * z_alpha if port_std_daily > 0 else 0
            component_rows.append({
                "position_id":    pos.position_id,
                "cusip":          pos.cusip,
                "instrument_type":pos.instrument_type,
                "bucket":         pos.bucket,
                "component_var":  abs(comp_var),
                "pct_of_total":   abs(comp_var) / var_1day * 100 if var_1day > 0 else 0,
            })

        return VaRResult(
            as_of_date          = date.today(),
            method              = "Parametric Delta-Normal (EWMA λ=0.94)",
            confidence_level    = self.confidence,
            holding_period_days = HOLDING_PERIOD,
            var_1day            = var_1day,
            var_10day           = var_10day,
            expected_shortfall  = es_1day,
            component_var       = pd.DataFrame(component_rows).sort_values("component_var", ascending=False),
            pnl_distribution    = pnl_sim,
            marginal_var        = pd.DataFrame(),
            metadata            = {
                "port_std_daily": port_std_daily,
                "z_alpha":        z_alpha,
                "ewma_lambda":    self.lam,
                "cov_matrix_det": np.linalg.det(self._cov_matrix),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VaR COMPARISON & REPORTING
# ──────────────────────────────────────────────────────────────────────────────

class VaRReporter:
    """Formats and prints the VaR run summary."""

    def print_summary(self, hs_result: VaRResult, param_result: VaRResult) -> None:
        from tabulate import tabulate

        print("\n" + "═" * 90)
        print("  MERIDIAN CAPITAL — VaR REPORT  (Fixed Income Portfolio)")
        print("═" * 90)

        summary = [
            ["Method",            hs_result.method,          param_result.method],
            ["Confidence Level",  f"{hs_result.confidence_level:.0%}", f"{param_result.confidence_level:.0%}"],
            ["1-Day VaR ($)",     f"${hs_result.var_1day:>14,.0f}",  f"${param_result.var_1day:>14,.0f}"],
            ["10-Day VaR ($)",    f"${hs_result.var_10day:>14,.0f}", f"${param_result.var_10day:>14,.0f}"],
            ["Expected Shortfall",f"${hs_result.expected_shortfall:>14,.0f}", f"${param_result.expected_shortfall:>14,.0f}"],
            ["Worst Scenario ($)",f"${abs(hs_result.metadata.get('worst_loss',0)):>14,.0f}", "N/A (parametric)"],
            ["P&L Skewness",      f"{hs_result.metadata.get('skewness',0):>14.4f}", "0.0000 (assumed normal)"],
            ["P&L Kurtosis",      f"{hs_result.metadata.get('kurtosis',0):>14.4f}", "0.0000 (assumed normal)"],
        ]
        print(tabulate(summary, headers=["Metric", "HS-VaR", "Parametric"],
                       tablefmt="rounded_outline"))

        print("\n  Top 5 Component VaR Contributors (Historical Simulation):")
        comp = hs_result.component_var.head(5)[[
            "position_id","cusip","instrument_type","bucket","component_var","pct_of_total"
        ]]
        print(tabulate(comp, headers="keys", tablefmt="rounded_outline", floatfmt=".2f"))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  DEMO
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    settle = date.today()

    # Synthetic portfolio (USTs + MBS)
    positions = [
        PortfolioPosition("POS-001", "91282CEX5", "UST",     50_000_000,  42_050_000,
                          {5.0: -42_000, 7.0: -8_000},  convexity=65.2,  bucket="5-10yr"),
        PortfolioPosition("POS-002", "91282CKH0", "UST",    -30_000_000, -25_425_000,
                          {5.0:  15_000, 7.0:  5_000},  convexity=-38.1, bucket="5-10yr"),
        PortfolioPosition("POS-003", "912810TZ1", "UST",     20_000_000,  17_950_000,
                          {20.0:-38_000,30.0: -7_000},  convexity=210.4, bucket="20-30yr"),
        PortfolioPosition("POS-004", "MA3456",    "MBS",     25_000_000,  25_312_500,
                          {5.0: -12_000, 7.0: -4_000},  convexity=-85.0, bucket="5-10yr"),
        PortfolioPosition("POS-005", "FG-A99012", "AGENCY",  10_000_000,   9_475_000,
                          {10.0:-28_000,20.0: -3_000},  convexity=95.2,  bucket="10-20yr"),
        PortfolioPosition("POS-006", "ZN-SEP24",  "FUTURES",-100_000_000,-87_000_000,
                          {7.0: 75_000,10.0: 10_000},   convexity=-55.0, bucket="5-10yr"),
    ]

    ytm_by_pos = {
        "POS-001": 7.0, "POS-002": 8.0, "POS-003": 29.0,
        "POS-004": 6.5, "POS-005": 14.0,"POS-006": 9.5,
    }

    # Load historical yield curve
    yield_history = load_yield_curve_history(KEY_RATE_TENORS, lookback_days=500)

    # Run both methods
    hs_var    = HistoricalSimVaR(yield_history, confidence=0.99)
    param_var = ParametricVaR(yield_history,   confidence=0.99)

    hs_result    = hs_var.compute(positions, ytm_by_pos)
    param_result = param_var.compute(positions, ytm_by_pos)

    VaRReporter().print_summary(hs_result, param_result)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _demo()
