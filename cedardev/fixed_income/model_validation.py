"""
model_validation.py
===================
VaR Model Validation & Performance Reporting
Firm: CedarDev Capital Management LLC  |  Desk: Market Risk Infrastructure
Version: 1.5.0  |  Python 3.11+

This module addresses the core responsibility of the role:
  "Design and produce model performance metrics and reports to support
   communications with both internal model users and external supervisors."

Validation Suite
----------------
  1. Kupiec POF Test (Proportion of Failures)
       - H₀: observed exception rate = expected rate (1 − confidence level)
       - LR statistic is chi-squared (df=1)
       - Standard academic and regulatory test

  2. Christoffersen Independence Test
       - Tests whether VaR exceptions cluster (violation of i.i.d. assumption)
       - Two exceptions on consecutive days suggests model underestimates
         autocorrelation in volatility

  3. Basel Traffic Light
       - Green  (0–4 exceptions in 250 days)  : internal model accepted
       - Yellow (5–9 exceptions)              : supervisory review required
       - Red    (10+ exceptions)              : model rejected, capital add-on

  4. P&L Attribution (PLA) Test  — FRTB requirement
       - Compares model P&L (risk-factor) to actual P&L
       - Spearman correlation and mean ratio must both be within bounds

  5. Model Sensitivity Report  (what SEC auditors actually look at)
       - VaR vs lookback window length
       - VaR vs confidence level
       - VaR vs EWMA lambda
       - VaR vs PSA speed assumption (for MBS in the portfolio)

References
----------
  - Basel Committee, "Supervisory Framework for the Use of Backtesting" (1996)
  - Christoffersen, "Evaluating Interval Forecasts" (1998)
  - FRTB (Basel Jan 2019), Section "P&L Attribution"
  - SEC Rule 15c3-1 — net capital requirements
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestWindow:
    """A daily pair of (VaR estimate, realized P&L) for one holding period."""
    date:           date
    var_estimate:   float    # positive: the amount you could lose
    actual_pnl:     float    # realized P&L (negative = loss)
    is_exception:   bool     = field(init=False)

    def __post_init__(self):
        self.is_exception = self.actual_pnl < -self.var_estimate


@dataclass
class KupiecResult:
    """Output of the Kupiec POF test."""
    n_observations:     int
    n_exceptions:       int
    expected_rate:      float
    observed_rate:      float
    lr_statistic:       float
    p_value:            float
    reject_h0:          bool     # True = model is miscalibrated
    confidence_level:   float


@dataclass
class ChristoffersenResult:
    """Output of the Christoffersen independence test."""
    n_00: int    # non-exception followed by non-exception
    n_01: int    # non-exception followed by exception
    n_10: int    # exception followed by non-exception
    n_11: int    # exception followed by exception
    lr_independence:  float
    lr_coverage:      float
    lr_joint:         float
    p_value_joint:    float
    reject_independence: bool


@dataclass
class BaselTrafficLight:
    """Basel Committee backtesting zone assignment."""
    n_exceptions: int
    zone:         str     # "Green", "Yellow", "Red"
    plus_factor:  float   # additional capital multiplier (0 for Green)
    message:      str


@dataclass
class ValidationReport:
    """Comprehensive model validation output."""
    as_of_date:              date
    lookback_days:           int
    confidence_level:        float
    kupiec:                  KupiecResult
    christoffersen:          ChristoffersenResult
    traffic_light:           BaselTrafficLight
    pla_spearman_corr:       float
    pla_mean_ratio:          float
    pla_passed:              bool
    sensitivity_table:       pd.DataFrame
    passed_overall:          bool
    summary_notes:           List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  KUPIEC PROPORTION OF FAILURES TEST
# ──────────────────────────────────────────────────────────────────────────────

class KupiecTest:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    LR statistic (under H₀: p̂ = p):
        LR = −2 × ln[ p^x × (1−p)^(n−x) / (p̂^x × (1−p̂)^(n−x)) ]
    where:
        p  = expected exception rate  (1 − confidence level)
        p̂  = observed exception rate  (x/n)
        x  = number of exceptions
        n  = number of observations
        LR ~ χ²(1) under H₀
    """

    def run(
        self,
        observations:     List[BacktestWindow],
        confidence_level: float = 0.99,
    ) -> KupiecResult:
        n = len(observations)
        x = sum(1 for o in observations if o.is_exception)
        p_expected = 1 - confidence_level   # e.g. 0.01 at 99%
        p_hat      = x / n if n > 0 else 0

        if p_hat == 0:
            log.warning("Zero exceptions observed. LR statistic undefined; setting to 0.")
            lr_stat = 0.0
        elif p_hat == 1:
            log.warning("All observations are exceptions. Model is catastrophically wrong.")
            lr_stat = 1e10
        else:
            log._null = None   # suppress false reference
            # Likelihood under H₀ (model correct)
            ll_null = x * math.log(p_expected) + (n - x) * math.log(1 - p_expected)
            # Likelihood under H₁ (unconstrained)
            ll_alt  = x * math.log(p_hat)      + (n - x) * math.log(1 - p_hat)
            lr_stat = -2 * (ll_null - ll_alt)

        p_value  = 1 - stats.chi2.cdf(lr_stat, df=1)
        reject   = p_value < 0.05   # 5% significance level

        log.info(
            "Kupiec POF: n=%d, exceptions=%d (expected=%.1f), "
            "LR=%.4f, p=%.4f, %s",
            n, x, n * p_expected, lr_stat, p_value,
            "REJECT H0 (miscalibrated)" if reject else "FAIL TO REJECT (OK)"
        )

        return KupiecResult(
            n_observations   = n,
            n_exceptions     = x,
            expected_rate    = p_expected,
            observed_rate    = p_hat,
            lr_statistic     = lr_stat,
            p_value          = p_value,
            reject_h0        = reject,
            confidence_level = confidence_level,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CHRISTOFFERSEN INDEPENDENCE TEST
# ──────────────────────────────────────────────────────────────────────────────

class ChristoffersenTest:
    """
    Christoffersen (1998) conditional coverage test.

    Tests whether exceptions cluster (i.e., the probability of an exception
    on day t depends on whether day t−1 was also an exception).  A good VaR
    model should have i.i.d. exceptions — no clustering.

    LR_independence ~ χ²(1)
    LR_joint (coverage + independence) ~ χ²(2)
    """

    def run(self, observations: List[BacktestWindow]) -> ChristoffersenResult:
        hits = [int(o.is_exception) for o in observations]
        n    = len(hits)

        n_00 = n_01 = n_10 = n_11 = 0
        for i in range(1, n):
            prev, curr = hits[i-1], hits[i]
            if   prev == 0 and curr == 0: n_00 += 1
            elif prev == 0 and curr == 1: n_01 += 1
            elif prev == 1 and curr == 0: n_10 += 1
            elif prev == 1 and curr == 1: n_11 += 1

        pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        pi    = (n_01 + n_11) / (n - 1) if n > 1 else 0

        def safe_log(x, default=-1e10):
            return math.log(x) if x > 0 else default

        # Independence LR: H₀ = π₀₁ = π₁₁ (no clustering)
        ll_indep = (
            (n_00 + n_10) * safe_log(1 - pi)
            + (n_01 + n_11) * safe_log(pi)
        )
        ll_dep = (
            n_00 * safe_log(1 - pi_01) + n_01 * safe_log(pi_01)
            + n_10 * safe_log(1 - pi_11) + n_11 * safe_log(pi_11)
        )
        lr_ind = -2 * (ll_indep - ll_dep)

        # Coverage LR (Kupiec)
        p_expected = 0.01   # 99% VaR
        x = n_01 + n_11
        ll_cov_null = x * safe_log(p_expected) + (n-1-x) * safe_log(1-p_expected)
        p_hat       = x / (n - 1) if n > 1 else 0
        ll_cov_alt  = x * safe_log(p_hat) + (n-1-x) * safe_log(1-p_hat) if 0 < p_hat < 1 else 0
        lr_cov      = -2 * (ll_cov_null - ll_cov_alt)

        lr_joint  = lr_cov + lr_ind
        p_val_jnt = 1 - stats.chi2.cdf(lr_joint, df=2)
        reject_ind = p_val_jnt < 0.05

        log.info(
            "Christoffersen: n_00=%d n_01=%d n_10=%d n_11=%d  "
            "LR_joint=%.4f p=%.4f  %s",
            n_00, n_01, n_10, n_11, lr_joint, p_val_jnt,
            "CLUSTERING DETECTED" if reject_ind else "Independence OK"
        )

        return ChristoffersenResult(
            n_00=n_00, n_01=n_01, n_10=n_10, n_11=n_11,
            lr_independence  = lr_ind,
            lr_coverage      = lr_cov,
            lr_joint         = lr_joint,
            p_value_joint    = p_val_jnt,
            reject_independence = reject_ind,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  BASEL TRAFFIC LIGHT
# ──────────────────────────────────────────────────────────────────────────────

class BaselTrafficLightAssigner:
    """
    Assigns a Basel supervisory zone based on the number of backtesting
    exceptions over the most recent 250 trading days.

    Plus-factor (multiplier on VaR for capital purposes):
      Green  (0–4)  :  0.00
      Yellow (5)    :  0.40
      Yellow (6)    :  0.50
      Yellow (7)    :  0.65
      Yellow (8)    :  0.75
      Yellow (9)    :  0.85
      Red    (10+)  :  1.00  (model effectively rejected)
    """

    PLUS_FACTORS: Dict[int, float] = {
        0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00,
        5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85,
    }

    def assign(self, n_exceptions: int, n_observations: int = 250) -> BaselTrafficLight:
        if n_exceptions <= 4:
            zone, msg = "Green",  "Internal model accepted — no supervisory action."
        elif n_exceptions <= 9:
            zone, msg = "Yellow", f"Supervisory review required — model under scrutiny."
        else:
            zone, msg = "Red",    "Model rejected — switch to standardized approach."

        plus_factor = self.PLUS_FACTORS.get(
            min(n_exceptions, 9), 1.00
        )

        log.info(
            "Basel Traffic Light: %d exceptions/%d days → %s zone (plus=%.2f).",
            n_exceptions, n_observations, zone, plus_factor
        )

        return BaselTrafficLight(
            n_exceptions = n_exceptions,
            zone         = zone,
            plus_factor  = plus_factor,
            message      = msg,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  P&L ATTRIBUTION TEST  (FRTB requirement)
# ──────────────────────────────────────────────────────────────────────────────

class PLATest:
    """
    FRTB P&L Attribution (PLA) Test.

    Compares the "hypothetical P&L" (model-computed using risk factors only)
    to the "actual P&L" (desk P&L including all effects).

    Pass criteria (FRTB):
      1. Spearman correlation between HPL and APL ≥ 0.80
      2. |mean(HPL / APL) − 1| ≤ 0.10  (mean ratio within 10% of 1)

    Failure → desk uses standardized approach (higher capital charge).
    """

    SPEARMAN_THRESHOLD = 0.80
    MEAN_RATIO_BOUND   = 0.10

    def run(
        self,
        hypothetical_pnl: np.ndarray,   # model P&L (from risk factors)
        actual_pnl:       np.ndarray,   # desk P&L
    ) -> Tuple[float, float, bool]:
        """Returns (spearman_corr, mean_ratio, passed)."""
        if len(hypothetical_pnl) != len(actual_pnl):
            raise ValueError("HPL and APL arrays must have the same length.")

        spearman, _ = stats.spearmanr(hypothetical_pnl, actual_pnl)

        # Mean ratio: avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios    = np.where(actual_pnl != 0, hypothetical_pnl / actual_pnl, np.nan)
            mean_ratio = float(np.nanmean(ratios))

        passed = (
            spearman   >= self.SPEARMAN_THRESHOLD
            and abs(mean_ratio - 1.0) <= self.MEAN_RATIO_BOUND
        )
        log.info(
            "PLA Test: Spearman=%.4f (threshold=%.2f), MeanRatio=%.4f → %s",
            spearman, self.SPEARMAN_THRESHOLD, mean_ratio,
            "PASS" if passed else "FAIL"
        )
        return spearman, mean_ratio, passed


# ──────────────────────────────────────────────────────────────────────────────
# 6.  SENSITIVITY ANALYSIS  (model robustness documentation)
# ──────────────────────────────────────────────────────────────────────────────

def var_sensitivity_table(
    hs_var_func,     # callable(lookback, confidence) → VaRResult
    param_var_func,  # callable(confidence, lambda) → VaRResult
) -> pd.DataFrame:
    """
    Sweeps key model parameters and records 1-day VaR.
    Produces the sensitivity table required in SEC model documentation.
    """
    rows = []

    # Sweep 1: Lookback window (Historical Simulation)
    for lb in [125, 250, 375, 500, 750]:
        try:
            r = hs_var_func(lb, 0.99)
            rows.append({"parameter": f"HS lookback={lb}d", "var_1day": r.var_1day,
                          "method": "HS-VaR", "value": lb})
        except Exception as e:
            log.warning("Sensitivity sweep failed for lookback=%d: %s", lb, e)

    # Sweep 2: Confidence level
    for conf in [0.95, 0.97, 0.99, 0.999]:
        try:
            r = hs_var_func(500, conf)
            rows.append({"parameter": f"HS conf={conf:.1%}", "var_1day": r.var_1day,
                          "method": "HS-VaR", "value": conf})
        except Exception as e:
            log.warning("Sensitivity sweep failed for conf=%.3f: %s", conf, e)

    # Sweep 3: EWMA lambda (Parametric)
    for lam in [0.90, 0.92, 0.94, 0.96, 0.98]:
        try:
            r = param_var_func(0.99, lam)
            rows.append({"parameter": f"Param λ={lam}", "var_1day": r.var_1day,
                          "method": "Parametric", "value": lam})
        except Exception as e:
            log.warning("Sensitivity sweep failed for lambda=%.2f: %s", lam, e)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

class ModelValidator:
    """
    Full model validation orchestrator.

    Runs all tests and produces the consolidated ValidationReport that is:
      1. Filed with the SEC (external supervisor) on a monthly basis.
      2. Reviewed by internal model governance committee (quarterly).
      3. Used by the desk to monitor model drift continuously.
    """

    def run(
        self,
        backtest_data:    List[BacktestWindow],
        hypothetical_pnl: np.ndarray,
        actual_pnl:       np.ndarray,
        confidence_level: float = 0.99,
        sensitivity_df:   Optional[pd.DataFrame] = None,
    ) -> ValidationReport:
        log.info("=" * 60)
        log.info("  Model Validation Suite — starting")
        log.info("=" * 60)

        # 1. Kupiec
        kupiec  = KupiecTest().run(backtest_data, confidence_level)

        # 2. Christoffersen
        christoffersen = ChristoffersenTest().run(backtest_data)

        # 3. Basel Traffic Light (uses last 250 days per Basel spec)
        last_250    = backtest_data[-250:] if len(backtest_data) >= 250 else backtest_data
        n_exc_250   = sum(1 for o in last_250 if o.is_exception)
        traffic_light = BaselTrafficLightAssigner().assign(n_exc_250, len(last_250))

        # 4. PLA
        spearman, mean_ratio, pla_passed = PLATest().run(hypothetical_pnl, actual_pnl)

        # 5. Overall pass/fail
        passed_overall = (
            not kupiec.reject_h0
            and not christoffersen.reject_independence
            and traffic_light.zone in ("Green", "Yellow")
            and pla_passed
        )

        notes = []
        if kupiec.reject_h0:
            notes.append(
                f"⚠  Kupiec POF FAILED: observed={kupiec.observed_rate:.4f}, "
                f"expected={kupiec.expected_rate:.4f}."
            )
        if christoffersen.reject_independence:
            notes.append(
                f"⚠  Exception clustering detected: {christoffersen.n_11} "
                "consecutive-day exception pairs."
            )
        if traffic_light.zone == "Red":
            notes.append("🔴  Basel RED zone — model must be replaced immediately.")
        elif traffic_light.zone == "Yellow":
            notes.append(f"🟡  Basel YELLOW zone — {traffic_light.message}")
        if not pla_passed:
            notes.append(
                f"⚠  PLA Test FAILED: Spearman={spearman:.4f}, "
                f"MeanRatio={mean_ratio:.4f}."
            )
        if passed_overall:
            notes.append("✅  All validation checks passed. Model approved for production use.")

        return ValidationReport(
            as_of_date          = date.today(),
            lookback_days       = len(backtest_data),
            confidence_level    = confidence_level,
            kupiec              = kupiec,
            christoffersen      = christoffersen,
            traffic_light       = traffic_light,
            pla_spearman_corr   = spearman,
            pla_mean_ratio      = mean_ratio,
            pla_passed          = pla_passed,
            sensitivity_table   = sensitivity_df or pd.DataFrame(),
            passed_overall      = passed_overall,
            summary_notes       = notes,
        )

    def print_report(self, r: ValidationReport) -> None:
        DIVIDER = "─" * 90

        print(f"\n{'═'*90}")
        print("  CedarDev Capital — MODEL VALIDATION REPORT")
        print(f"  As of: {r.as_of_date}  |  Lookback: {r.lookback_days} days  "
              f"|  Confidence: {r.confidence_level:.0%}")
        print(f"{'═'*90}")

        # Kupiec
        print(f"\n{DIVIDER}")
        print("  KUPIEC PROPORTION OF FAILURES TEST")
        print(DIVIDER)
        kupiec_data = [
            ["Observations",        r.kupiec.n_observations],
            ["Exceptions",          r.kupiec.n_exceptions],
            ["Expected exceptions", f"{r.kupiec.n_observations * r.kupiec.expected_rate:.1f}"],
            ["Observed rate",       f"{r.kupiec.observed_rate:.4f}"],
            ["Expected rate",       f"{r.kupiec.expected_rate:.4f}"],
            ["LR statistic",        f"{r.kupiec.lr_statistic:.4f}"],
            ["p-value",             f"{r.kupiec.p_value:.4f}"],
            ["Result",              "REJECT (miscalibrated)" if r.kupiec.reject_h0 else "PASS"],
        ]
        print(tabulate(kupiec_data, tablefmt="rounded_outline"))

        # Christoffersen
        print(f"\n{DIVIDER}")
        print("  CHRISTOFFERSEN INDEPENDENCE TEST")
        print(DIVIDER)
        c = r.christoffersen
        c_data = [
            ["Transition counts", f"n00={c.n_00} n01={c.n_01} n10={c.n_10} n11={c.n_11}"],
            ["LR joint",          f"{c.lr_joint:.4f}"],
            ["p-value (joint)",   f"{c.p_value_joint:.4f}"],
            ["Clustering",        "DETECTED" if c.reject_independence else "None (OK)"],
        ]
        print(tabulate(c_data, tablefmt="rounded_outline"))

        # Traffic Light
        print(f"\n{DIVIDER}")
        print("  BASEL TRAFFIC LIGHT")
        print(DIVIDER)
        tl = r.traffic_light
        color_map = {"Green": "✅ Green", "Yellow": "🟡 Yellow", "Red": "🔴 Red"}
        tl_data = [
            ["Exceptions (last 250d)", tl.n_exceptions],
            ["Zone",                   color_map.get(tl.zone, tl.zone)],
            ["Capital plus-factor",    f"+{tl.plus_factor:.2f}"],
            ["Action",                 tl.message],
        ]
        print(tabulate(tl_data, tablefmt="rounded_outline"))

        # PLA
        print(f"\n{DIVIDER}")
        print("  P&L ATTRIBUTION TEST (FRTB)")
        print(DIVIDER)
        pla_data = [
            ["Spearman correlation",  f"{r.pla_spearman_corr:.4f}  (threshold ≥ 0.80)"],
            ["Mean HPL/APL ratio",    f"{r.pla_mean_ratio:.4f}   (|ratio−1| ≤ 0.10)"],
            ["Result",                "PASS" if r.pla_passed else "FAIL"],
        ]
        print(tabulate(pla_data, tablefmt="rounded_outline"))

        # Sensitivity
        if not r.sensitivity_table.empty:
            print(f"\n{DIVIDER}")
            print("  SENSITIVITY ANALYSIS (Model Robustness)")
            print(DIVIDER)
            print(tabulate(r.sensitivity_table, headers="keys",
                           tablefmt="rounded_outline", floatfmt=".2f"))

        # Summary
        print(f"\n{DIVIDER}")
        print("  OVERALL ASSESSMENT")
        print(DIVIDER)
        for note in r.summary_notes:
            print(f"  {note}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  DEMO
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    np.random.seed(42)
    n = 500

    # Simulate a well-behaved model: 99% VaR, ~1% exception rate, no clustering
    true_vol    = 500_000     # daily P&L volatility ($)
    var_scale   = 2.326       # z_99
    var_series  = np.full(n, true_vol * var_scale)

    actual_pnl  = np.random.normal(0, true_vol, n)
    # Add mild fat tails (5% of days draw from a heavier distribution)
    fat_days    = np.random.choice(n, size=int(n * 0.05), replace=False)
    actual_pnl[fat_days] *= np.random.uniform(1.5, 3.0, len(fat_days))

    backtest_data = [
        BacktestWindow(
            date          = date(2023, 1, 1) + timedelta(days=i),
            var_estimate  = var_series[i],
            actual_pnl    = actual_pnl[i],
        )
        for i in range(n)
    ]

    # Hypothetical vs actual P&L (for PLA test)
    noise_ratio  = 0.05   # model explains 95% of P&L
    hypo_pnl     = actual_pnl * (1 - noise_ratio) + np.random.normal(0, true_vol * noise_ratio, n)

    validator = ModelValidator()
    report    = validator.run(
        backtest_data    = backtest_data,
        hypothetical_pnl = hypo_pnl,
        actual_pnl       = actual_pnl,
        confidence_level = 0.99,
    )
    validator.print_report(report)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _demo()
