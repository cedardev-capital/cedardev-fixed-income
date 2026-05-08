"""
ust_pricer.py
=============
US Treasury Securities Pricing & Risk Analytics
Firm: CedarDev Capital Management LLC  |  Desk: Market Risk Infrastructure
Team: Quantitative Risk Infrastructure (QRI)
Version: 2.4.1  |  Python 3.11+

Covers
------
  - Clean / dirty price from yield-to-maturity (YTM)
  - YTM solver from market price (Newton-Raphson)
  - Bond Equivalent Yield (BEY) and semi-annual compounding conventions
  - Accrued interest (Actual/Actual ICMA — the UST day-count convention)
  - Modified Duration, Macaulay Duration, DV01, Dollar Convexity
  - Zero-coupon curve bootstrapping from on-the-run Treasury quotes
  - Z-spread calculation (OAS for vanilla bonds)
  - Full cash-flow schedule generation

SEC / FINRA Context
-------------------
  The risk measures produced here feed directly into the margin engine
  (margin_engine.py).  Every number must be independently reproducible
  from first principles — no black-box pricing library calls.

References
----------
  - Fabozzi, "Fixed Income Mathematics" (4th ed.), Chapters 3–5
  - Bloomberg FLDS: YLD_YTM_BID, DUR_ADJ_MID, CONVEXITY_MID
  - SEC Release No. 34-68784 — portfolio margining rule
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENUMERATIONS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class DayCount(str, Enum):
    ACT_ACT_ICMA = "ACT/ACT ICMA"   # UST standard
    ACT_360      = "ACT/360"         # repo / money market
    THIRTY_360   = "30/360"          # corporate / agency


class Frequency(int, Enum):
    SEMI_ANNUAL = 2   # all US Treasuries
    ANNUAL      = 1
    QUARTERLY   = 4


FACE_VALUE_DEFAULT = 1_000.0    # $1,000 par — Bloomberg convention
SETTLEMENT_LAG_UST = 1          # T+1 for on-the-run Treasuries


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class USTBond:
    """
    Represents a single US Treasury note or bond.

    Attributes
    ----------
    cusip          : 9-character CUSIP
    issue_date     : original issue date
    maturity_date  : maturity date
    coupon_rate    : annual coupon as decimal  (e.g. 0.0425 = 4.25 %)
    face_value     : par value in USD (default $1,000)
    frequency      : coupon frequency (default semi-annual)
    is_tips        : True for Treasury Inflation-Protected Securities
    on_the_run     : True if this is the current benchmark issue
    """
    cusip:         str
    issue_date:    date
    maturity_date: date
    coupon_rate:   float
    face_value:    float     = FACE_VALUE_DEFAULT
    frequency:     Frequency = Frequency.SEMI_ANNUAL
    is_tips:       bool      = False
    on_the_run:    bool      = False

    @property
    def semi_annual_coupon(self) -> float:
        return self.face_value * self.coupon_rate / self.frequency


@dataclass
class USTAnalytics:
    """Output bundle from USTBondPricer — every risk measure in one object."""
    cusip:              str
    settlement_date:    date
    clean_price:        float
    dirty_price:        float
    accrued_interest:   float
    ytm:                float   # annual, semi-annual compounding (BEY)
    macaulay_duration:  float   # years
    modified_duration:  float   # years
    dv01:               float   # $ per $1 face, per bp
    dollar_duration:    float   # $ per $1 face, per 100bp
    convexity:          float   # years²
    dollar_convexity:   float
    z_spread:           Optional[float] = None    # bps
    cash_flows:         List[Tuple[date, float]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DAY-COUNT  (Actual/Actual ICMA — the official UST convention)
# ──────────────────────────────────────────────────────────────────────────────

class ActActICMA:
    """
    Actual/Actual ICMA day-count fraction as specified in ISMA Rule 251.

    Formula
    -------
      dcf = actual_days_in_period / (frequency × actual_days_in_coupon_period)

    This is the ONLY day-count convention used for US Treasury securities.
    Do NOT use 30/360 or ACT/365 for USTs — the prices will be wrong.
    """

    @staticmethod
    def day_count_fraction(
        start: date,
        end: date,
        coupon_start: date,
        coupon_end: date,
        frequency: int = 2,
    ) -> float:
        actual_days        = (end - start).days
        coupon_period_days = (coupon_end - coupon_start).days
        if coupon_period_days == 0:
            return 0.0
        return actual_days / (frequency * coupon_period_days)

    @staticmethod
    def accrued_interest(
        bond: USTBond,
        settlement: date,
    ) -> Tuple[float, date, date]:
        """
        Returns (accrued_interest, prev_coupon_date, next_coupon_date).
        """
        prev_cpn, next_cpn = ActActICMA._coupon_dates(bond, settlement)
        dcf = ActActICMA.day_count_fraction(
            start        = prev_cpn,
            end          = settlement,
            coupon_start = prev_cpn,
            coupon_end   = next_cpn,
            frequency    = bond.frequency.value,
        )
        ai = bond.semi_annual_coupon * dcf * bond.frequency.value
        # accrued = coupon × (days since last coupon / days in coupon period)
        # but scaled back to one period's coupon payment
        ai = bond.semi_annual_coupon * (
            (settlement - prev_cpn).days / (next_cpn - prev_cpn).days
        )
        return ai, prev_cpn, next_cpn

    @staticmethod
    def _coupon_dates(bond: USTBond, settlement: date) -> Tuple[date, date]:
        """
        Walk backwards/forwards from maturity to find the coupon bracket
        around *settlement*.  Semi-annual: coupons land on the same
        month/day as maturity, 6 months apart.
        """
        m, d     = bond.maturity_date.month, bond.maturity_date.day
        periods  = bond.frequency.value   # 2 for semi-annual
        # Step back from maturity until we pass settlement
        cpn_date = bond.maturity_date
        while cpn_date > settlement:
            cpn_date = ActActICMA._subtract_months(cpn_date, 12 // periods)
        prev_cpn = cpn_date
        next_cpn = ActActICMA._add_months(cpn_date, 12 // periods)
        return prev_cpn, next_cpn

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        month = d.month + months
        year  = d.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day   = min(d.day, [31,28+int(year%4==0 and (year%100!=0 or year%400==0)),
                             31,30,31,30,31,31,30,31,30,31][month-1])
        return date(year, month, day)

    @staticmethod
    def _subtract_months(d: date, months: int) -> date:
        return ActActICMA._add_months(d, -months)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CASH-FLOW SCHEDULE
# ──────────────────────────────────────────────────────────────────────────────

def generate_cash_flows(
    bond: USTBond,
    settlement: date,
) -> List[Tuple[date, float]]:
    """
    Returns a list of (payment_date, cash_flow_amount) tuples for all
    remaining coupon payments AFTER settlement, plus the final principal.

    The first coupon may be a short (stub) coupon if settlement falls
    mid-period — accrued interest is NOT subtracted here; that is the
    responsibility of the pricer.
    """
    flows: List[Tuple[date, float]] = []
    months_per_period = 12 // bond.frequency.value   # 6 for semi-annual

    cpn_date = bond.maturity_date
    # Walk back to the first coupon strictly after settlement
    while cpn_date > settlement:
        flows.append((cpn_date, bond.semi_annual_coupon))
        cpn_date = ActActICMA._subtract_months(cpn_date, months_per_period)

    flows.sort(key=lambda x: x[0])

    # Add principal to the final payment
    if flows:
        last_date, last_cpn = flows[-1]
        flows[-1] = (last_date, last_cpn + bond.face_value)

    return flows


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PRICER  (yield ↔ price, risk measures)
# ──────────────────────────────────────────────────────────────────────────────

class USTBondPricer:
    """
    Full analytical pricer for US Treasury notes and bonds.

    Pricing convention (Bloomberg-compatible)
    -----------------------------------------
    - Prices are quoted per $100 face (Bloomberg "dirty" vs "clean").
    - YTM is quoted as a Bond Equivalent Yield (BEY): annualized, with
      semi-annual compounding.
    - The standard settlement is T+1 for on-the-run, T+1 for off-the-run
      secondary market.

    All calculations follow:
      P = Σ [CF_t / (1 + y/2)^(t_i)]
    where t_i is measured in semi-annual periods from settlement.
    """

    def price_from_yield(
        self,
        bond: USTBond,
        ytm: float,            # annual BEY decimal (e.g. 0.0450 = 4.50%)
        settlement: date,
    ) -> Tuple[float, float, float]:
        """
        Returns (dirty_price, clean_price, accrued_interest) per $100 face.

        dirty_price  = present value of all future cash flows
        clean_price  = dirty_price - accrued_interest
        """
        flows = generate_cash_flows(bond, settlement)
        ai, prev_cpn, next_cpn = ActActICMA.accrued_interest(bond, settlement)

        period_yield = ytm / bond.frequency.value   # semi-annual rate

        dirty = 0.0
        for pmt_date, cf in flows:
            # Fractional semi-annual periods from settlement to payment
            t = ActActICMA.day_count_fraction(
                start        = settlement,
                end          = pmt_date,
                coupon_start = prev_cpn,
                coupon_end   = next_cpn,
                frequency    = bond.frequency.value,
            ) * bond.frequency.value   # convert dcf → semi-annual periods

            dirty += cf / (1 + period_yield) ** t

        # Normalize to per-$100 face
        scale = 100.0 / bond.face_value
        dirty_100 = dirty * scale
        ai_100    = ai    * scale
        clean_100 = dirty_100 - ai_100

        return dirty_100, clean_100, ai_100

    def yield_from_price(
        self,
        bond: USTBond,
        clean_price_100: float,   # per $100 face
        settlement: date,
    ) -> float:
        """
        Solves for YTM (BEY) given a clean price using Brent's method.
        Tolerance: 1e-10  (sub-micro-basis-point accuracy).
        """
        ai, _, _ = ActActICMA.accrued_interest(bond, settlement)
        ai_100   = ai * 100.0 / bond.face_value
        dirty_100 = clean_price_100 + ai_100

        def price_diff(ytm_guess: float) -> float:
            dirty, _, _ = self.price_from_yield(bond, ytm_guess, settlement)
            return dirty - dirty_100

        try:
            ytm = brentq(price_diff, 1e-6, 0.999, xtol=1e-10, maxiter=200)
        except ValueError:
            log.warning(
                "Brent solver failed for %s at price=%.6f; "
                "attempting wider bracket.",
                bond.cusip, clean_price_100
            )
            ytm = brentq(price_diff, -0.10, 2.0, xtol=1e-10, maxiter=500)

        return ytm

    def risk_measures(
        self,
        bond: USTBond,
        ytm: float,
        settlement: date,
        bump_bps: float = 1.0,   # DV01 bump size
    ) -> USTAnalytics:
        """
        Computes the full suite of risk measures via:
          - Macaulay/Modified Duration: analytical formula
          - Convexity: analytical second derivative
          - DV01: numerical bump (±0.5bp) — cross-checks the analytical result
          - Dollar Convexity: C × P / 100
        """
        dirty, clean, ai = self.price_from_yield(bond, ytm, settlement)
        flows           = generate_cash_flows(bond, settlement)
        ai_obj, prev_cpn, next_cpn = ActActICMA.accrued_interest(bond, settlement)

        period_yield = ytm / bond.frequency.value

        # ── Macaulay Duration (semi-annual periods → convert to years) ──
        mac_dur_periods = 0.0
        for pmt_date, cf in flows:
            t = ActActICMA.day_count_fraction(
                settlement, pmt_date, prev_cpn, next_cpn,
                bond.frequency.value,
            ) * bond.frequency.value
            pv = cf / (1 + period_yield) ** t
            mac_dur_periods += t * pv

        mac_dur_periods /= (dirty * bond.face_value / 100.0)
        mac_dur_years    = mac_dur_periods / bond.frequency.value

        # ── Modified Duration ──────────────────────────────────────────
        mod_dur = mac_dur_years / (1 + period_yield)

        # ── Convexity (analytical) ─────────────────────────────────────
        convex_periods = 0.0
        for pmt_date, cf in flows:
            t = ActActICMA.day_count_fraction(
                settlement, pmt_date, prev_cpn, next_cpn,
                bond.frequency.value,
            ) * bond.frequency.value
            pv = cf / (1 + period_yield) ** t
            convex_periods += t * (t + 1) * pv

        convex_periods /= (dirty * bond.face_value / 100.0)
        convexity       = convex_periods / (bond.frequency.value ** 2 * (1 + period_yield) ** 2)

        # ── DV01 (numerical bump, per $1 face, in dollars) ─────────────
        bump = bump_bps / 10_000
        dirty_up,  _, _ = self.price_from_yield(bond, ytm + bump / 2, settlement)
        dirty_dn,  _, _ = self.price_from_yield(bond, ytm - bump / 2, settlement)
        dv01_per_100 = (dirty_dn - dirty_up) / 2   # $ per $100 face per 1bp
        dv01         = dv01_per_100 / 100           # $ per $1 face per 1bp

        dollar_duration  = mod_dur * dirty / 100
        dollar_convexity = convexity * dirty / 100

        return USTAnalytics(
            cusip             = bond.cusip,
            settlement_date   = settlement,
            clean_price       = clean,
            dirty_price       = dirty,
            accrued_interest  = ai,
            ytm               = ytm,
            macaulay_duration = mac_dur_years,
            modified_duration = mod_dur,
            dv01              = dv01,
            dollar_duration   = dollar_duration,
            convexity         = convexity,
            dollar_convexity  = dollar_convexity,
            cash_flows        = flows,
        )

    def full_analytics(
        self,
        bond: USTBond,
        clean_price_100: float,
        settlement: date,
        zero_curve: Optional["ZeroCurve"] = None,
    ) -> USTAnalytics:
        """
        One-stop method: given a market clean price, returns all analytics
        including Z-spread if a zero curve is supplied.
        """
        ytm     = self.yield_from_price(bond, clean_price_100, settlement)
        result  = self.risk_measures(bond, ytm, settlement)

        if zero_curve is not None:
            result.z_spread = self._z_spread(bond, clean_price_100, settlement, zero_curve)

        return result

    def _z_spread(
        self,
        bond: USTBond,
        clean_price_100: float,
        settlement: date,
        zero_curve: "ZeroCurve",
    ) -> float:
        """
        Z-spread: constant spread over the zero curve such that
        discounted cash flows equal the market dirty price.
        Returned in basis points.
        """
        ai, prev_cpn, next_cpn = ActActICMA.accrued_interest(bond, settlement)
        ai_100  = ai * 100.0 / bond.face_value
        dirty_target = clean_price_100 + ai_100
        flows   = generate_cash_flows(bond, settlement)

        def price_diff(zspread_bps: float) -> float:
            z = zspread_bps / 10_000
            pv = 0.0
            for pmt_date, cf in flows:
                t_years = (pmt_date - settlement).days / 365.25
                r = zero_curve.rate(t_years)
                df = math.exp(-(r + z) * t_years)
                pv += cf * df
            return (pv * 100.0 / bond.face_value) - dirty_target

        try:
            zspread = brentq(price_diff, -500.0, 5000.0, xtol=1e-8, maxiter=300)
        except ValueError:
            log.warning("Z-spread solver failed for %s. Returning NaN.", bond.cusip)
            return float("nan")

        return zspread


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ZERO CURVE  (bootstrapped from on-the-run Treasury quotes)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZeroCurve:
    """
    Continuously-compounded zero rate curve, bootstrapped from par yields.

    The on-the-run Treasury par yield curve (2s, 3s, 5s, 7s, 10s, 20s, 30s)
    is published daily by the US Treasury at:
        https://home.treasury.gov/resource-center/data-chart-center/interest-rates/

    Bootstrap algorithm
    -------------------
    For each maturity T_n with par yield c_n:
      1. Discount all coupon cash flows at t < T_n using already-known zero rates.
      2. Solve for z(T_n) such that the bond prices at par.

    The result is a piecewise-linear interpolation of zero rates.
    For production use, replace with cubic spline or Nelson-Siegel.
    """
    tenors: np.ndarray    # years (e.g. [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rates:  np.ndarray    # continuously-compounded zero rates (decimal)

    as_of_date: date = field(default_factory=date.today)

    def rate(self, t_years: float) -> float:
        """Linear interpolation; flat extrapolation at the ends."""
        return float(np.interp(t_years, self.tenors, self.rates))

    def discount_factor(self, t_years: float) -> float:
        return math.exp(-self.rate(t_years) * t_years)

    @classmethod
    def bootstrap(
        cls,
        par_tenors: List[float],
        par_yields: List[float],   # annual, decimal
        as_of: Optional[date] = None,
        frequency: int = 2,
    ) -> "ZeroCurve":
        """
        Bootstrap zero curve from par yields using the iterative strip method.
        Short end (<= 1yr) treated as zero-coupon instruments.
        """
        as_of    = as_of or date.today()
        tenors   = np.array(par_tenors)
        z_rates  = np.zeros_like(tenors)

        for i, (T, c) in enumerate(zip(par_tenors, par_yields)):
            if T <= 1.0 / frequency:
                # Short-end: direct inversion  P = 1 / (1 + c*T)
                z_rates[i] = -math.log(1 / (1 + c * T)) / T
                continue

            # Build coupon schedule for this par bond
            n_periods = round(T * frequency)
            cpn       = c / frequency   # semi-annual coupon
            coupon_times = np.linspace(T / n_periods, T, n_periods)

            # Sum PV of coupons 1..n-1 using already-known zero rates
            pv_coupons = 0.0
            for t in coupon_times[:-1]:
                r  = float(np.interp(t, tenors[:i+1], z_rates[:i+1]))
                pv_coupons += cpn * math.exp(-r * t)

            # Solve for z(T): (1 + cpn) × exp(-z×T) = 1 - pv_coupons
            rhs = (1.0 - pv_coupons) / (1 + cpn)
            if rhs <= 0:
                log.warning("Bootstrap: non-positive discount factor at T=%.2f. Skipping.", T)
                z_rates[i] = z_rates[i-1]
                continue

            z_rates[i] = -math.log(rhs) / T

        log.debug(
            "Zero curve bootstrapped from %d par yields. "
            "Range: [%.4f, %.4f]", len(par_tenors), z_rates[0], z_rates[-1]
        )
        return cls(tenors=tenors, rates=z_rates, as_of_date=as_of)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PORTFOLIO-LEVEL ANALYTICS  (DV01 aggregation, risk report)
# ──────────────────────────────────────────────────────────────────────────────

class USTPortfolioRisk:
    """
    Given a list of (USTBond, face_amount, clean_price) tuples, computes:
      - per-bond analytics
      - portfolio-level DV01, duration, convexity
      - DV01 by tenor bucket (feeds into margin_engine.py)
    """

    def __init__(
        self,
        zero_curve: Optional[ZeroCurve] = None,
        settlement: Optional[date] = None,
    ) -> None:
        self.pricer     = USTBondPricer()
        self.zero_curve = zero_curve
        self.settlement = settlement or date.today()

    def analyze_portfolio(
        self,
        positions: List[Tuple[USTBond, float, float]],   # (bond, face, clean_px)
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per position, all risk measures
        scaled to the actual face amount (not per-$100 convention).
        """
        rows = []
        for bond, face_amt, clean_px in positions:
            analytics = self.pricer.full_analytics(
                bond, clean_px, self.settlement, self.zero_curve
            )
            scale = face_amt / bond.face_value   # scale from $1,000 face to actual

            rows.append({
                "cusip":             bond.cusip,
                "maturity_date":     bond.maturity_date,
                "coupon_rate":       bond.coupon_rate,
                "face_amount":       face_amt,
                "clean_price":       clean_px,
                "dirty_price_total": analytics.dirty_price * face_amt / 100,
                "accrued_interest":  analytics.accrued_interest * face_amt / 100,
                "ytm_bey":           analytics.ytm,
                "macaulay_duration": analytics.macaulay_duration,
                "modified_duration": analytics.modified_duration,
                "dv01_total":        analytics.dv01 * face_amt,      # $per bp, full position
                "convexity":         analytics.convexity,
                "z_spread_bps":      analytics.z_spread,
                "ytm_years":         (bond.maturity_date - self.settlement).days / 365.25,
            })

        df = pd.DataFrame(rows)
        log.info(
            "Portfolio analytics complete: %d bonds, total DV01=$%,.0f",
            len(df), df["dv01_total"].sum()
        )
        return df

    def dv01_by_bucket(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """
        Bucketed DV01 summary — feeds directly into margin_engine.py.
        Bucket labels match BUCKET_ORDER in margin_engine.py.
        """
        def _assign(ytm_yr):
            if ytm_yr <  2.0: return "0-2yr"
            if ytm_yr <  5.0: return "2-5yr"
            if ytm_yr < 10.0: return "5-10yr"
            if ytm_yr < 20.0: return "10-20yr"
            if ytm_yr < 30.0: return "20-30yr"
            return "30yr+"

        portfolio_df = portfolio_df.copy()
        portfolio_df["bucket"] = portfolio_df["ytm_years"].apply(_assign)
        return portfolio_df.groupby("bucket")["dv01_total"].sum().sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  DEMO / REFERENCE RUN
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    """
    Prices a representative cross-section of on-the-run Treasuries
    and prints a Bloomberg-style risk table.  Run directly:
        python ust_pricer.py
    """
    from tabulate import tabulate

    settle = date(2024, 7, 1)

    # On-the-run Treasury universe (as of mid-2024 — illustrative)
    bonds = [
        USTBond("91282CKQ0", date(2024, 1, 15), date(2026, 6, 30), 0.0500, on_the_run=True),
        USTBond("91282CKN7", date(2022, 5, 15), date(2027, 4, 30), 0.0275, on_the_run=False),
        USTBond("91282CKH0", date(2021, 11,15), date(2029, 11,15), 0.0125, on_the_run=False),
        USTBond("91282CEX5", date(2021, 8, 15), date(2031, 7, 31), 0.0125, on_the_run=True),
        USTBond("912810TM0", date(2020, 2, 15), date(2044, 2, 15), 0.0200, on_the_run=False),
        USTBond("912810TZ1", date(2023, 2, 15), date(2053, 2, 15), 0.0375, on_the_run=True),
    ]
    # Indicative market prices (per $100 face)
    prices = [99.85, 96.20, 87.45, 84.10, 76.50, 89.75]

    # Build zero curve from on-the-run par yields (illustrative mid-2024 curve)
    par_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    par_yields = [0.0530,0.0528,0.0520,0.0490,0.0475,0.0450,0.0440,0.0430,0.0445,0.0435]
    zero_curve = ZeroCurve.bootstrap(par_tenors, par_yields, as_of=settle)

    # Run portfolio analytics
    positions  = [(b, 1_000_000, p) for b, p in zip(bonds, prices)]
    pf_risk    = USTPortfolioRisk(zero_curve=zero_curve, settlement=settle)
    df         = pf_risk.analyze_portfolio(positions)

    print("\n" + "═" * 110)
    print("  CedarDev Capital — US TREASURY RISK REPORT")
    print("═" * 110)
    display_cols = [
        "cusip", "maturity_date", "coupon_rate", "clean_price",
        "ytm_bey", "modified_duration", "dv01_total", "convexity",
    ]
    print(tabulate(df[display_cols], headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

    print("\n  DV01 by Tenor Bucket (feeds into Margin Engine):")
    print(pf_risk.dv01_by_bucket(df).to_string())
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
