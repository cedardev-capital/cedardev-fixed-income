"""
mbs_analytics.py
================
Mortgage-Backed Securities (MBS) & Agency Analytics
Firm: CedarDev Capital Management LLC  |  Desk: Market Risk Infrastructure
Version: 1.9.2  |  Python 3.11+

Covers
------
  Agency MBS (Fannie Mae / Freddie Mac / Ginnie Mae pass-throughs)
  - PSA (Public Securities Association) prepayment model
  - CPR / SMM conversion (Conditional Prepayment Rate → Single Monthly Mortality)
  - Full monthly cash flow generation (scheduled principal + prepayment + interest)
  - Weighted Average Life (WAL)
  - Dollar price from OAS via numerical solver
  - DV01 and Modified Duration (via price bump)
  - Pool-level factor / paydown tracking

Regulatory Context
------------------
  MBS are included in FINRA Rule 4210 margining.  The SEC requires that
  margin calculations use the *model price* of the MBS (not just par),
  and that the prepayment assumption be documented and consistent with
  the SIFMA / Bloomberg median PSA speed.

References
----------
  - Fabozzi, "Mortgage-Backed Securities" (2nd ed.)
  - SIFMA Standard Prepayment Model (PSA)
  - Ginnie Mae MBS Guide, Chapter 18
  - Bloomberg FLDS: YLD_CNV_MID, WAL, OAD, OAS, PSA
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENUMERATIONS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

class AgencyType(str, Enum):
    FNMA  = "Fannie Mae"    # FNMA — conventional conforming
    FHLMC = "Freddie Mac"   # FHLMC — conventional conforming
    GNMA  = "Ginnie Mae"    # GNMA — FHA / VA government-backed


class CollateralType(str, Enum):
    FIXED_RATE  = "Fixed Rate"
    ARM         = "Adjustable Rate"
    HYBRID_ARM  = "Hybrid ARM"
    IO          = "Interest Only"


# 100% PSA benchmark: CPR ramps linearly from 0% to 6% over months 1–30,
# then remains flat at 6% for the life of the pool.
PSA_RAMP_MONTHS    = 30
PSA_PLATEAU_CPR    = 0.06     # 6% annual CPR at 100% PSA
MONTHS_PER_YEAR    = 12
SERVICING_SPREAD   = 0.0025   # typical 25 bps annual servicing fee

# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MBSPool:
    """
    Represents an agency MBS pool or TBA (To-Be-Announced) position.

    Attributes
    ----------
    pool_number       : e.g. "MA3456" (FNMA) or "G2 SF 30yr"
    agency            : FNMA / FHLMC / GNMA
    collateral_type   : Fixed rate, ARM, IO, etc.
    original_balance  : original pool balance at issuance ($)
    current_factor    : pool factor = current balance / original balance
    gross_coupon      : gross WAC (Weighted Average Coupon) annual decimal
    net_coupon        : pass-through coupon (gross − servicing)
    wam               : Weighted Average Maturity (months remaining)
    wala              : Weighted Average Loan Age (months)
    issue_date        : pool issuance date
    settlement_date   : settlement date for this analysis
    ltv               : Weighted Average LTV
    fico              : Weighted Average FICO
    """
    pool_number:       str
    agency:            AgencyType
    collateral_type:   CollateralType
    original_balance:  float
    current_factor:    float
    gross_coupon:      float
    net_coupon:        float
    wam:               int           # months
    wala:              int           # months
    issue_date:        date
    settlement_date:   date
    ltv:               float = 0.75
    fico:              int   = 720

    @property
    def current_balance(self) -> float:
        return self.original_balance * self.current_factor

    @property
    def remaining_term(self) -> int:
        """Remaining months = WAM (already reflects current seasoning)."""
        return self.wam

    @property
    def monthly_net_rate(self) -> float:
        return self.net_coupon / MONTHS_PER_YEAR


@dataclass
class MBSCashFlow:
    """Monthly cash flow for a single MBS period."""
    month:               int
    payment_date:        date
    beg_balance:         float
    scheduled_interest:  float
    scheduled_principal: float
    prepayment:          float
    total_principal:     float
    total_payment:       float
    end_balance:         float
    smm:                 float
    cpr:                 float


@dataclass
class MBSAnalytics:
    """Output bundle from MBSPricer."""
    pool_number:        str
    settlement_date:    date
    psa_speed:          float
    price_100:          float    # per $100 current face
    oas_bps:            float
    wal_years:          float
    modified_duration:  float
    dv01:               float    # $ per bp per $1 current face
    convexity:          float
    yield_to_maturity:  float
    cash_flows:         pd.DataFrame


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PSA PREPAYMENT MODEL
# ──────────────────────────────────────────────────────────────────────────────

class PSAModel:
    """
    Standard PSA (Public Securities Association) prepayment speed model.

    100% PSA benchmark
    ------------------
      Month 1 :  CPR = 6% × (1/30) = 0.20%
      Month 2 :  CPR = 6% × (2/30) = 0.40%
      ...
      Month 30+: CPR = 6.00%  (plateau)

    For n% PSA, multiply the benchmark CPR by n/100.

    CPR → SMM conversion
    --------------------
      SMM = 1 − (1 − CPR)^(1/12)
    """

    @staticmethod
    def cpr_schedule(
        psa_speed:      float,    # e.g. 150.0 = 150% PSA
        wala:           int,      # current loan age in months
        num_months:     int,      # number of months to project
    ) -> np.ndarray:
        """
        Returns array of CPR values (decimal) for each projection month.
        Takes into account the existing loan age (WALA) in the ramp.
        """
        cprs = np.zeros(num_months)
        for i in range(num_months):
            age = wala + i + 1   # loan age at end of month i
            if age <= PSA_RAMP_MONTHS:
                benchmark_cpr = PSA_PLATEAU_CPR * (age / PSA_RAMP_MONTHS)
            else:
                benchmark_cpr = PSA_PLATEAU_CPR
            cprs[i] = benchmark_cpr * (psa_speed / 100.0)
        return cprs

    @staticmethod
    def cpr_to_smm(cpr: float) -> float:
        """Single Monthly Mortality from annual CPR."""
        return 1.0 - (1.0 - cpr) ** (1.0 / MONTHS_PER_YEAR)

    @staticmethod
    def smm_to_cpr(smm: float) -> float:
        return 1.0 - (1.0 - smm) ** MONTHS_PER_YEAR


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CASH FLOW GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class MBSCashFlowGenerator:
    """
    Generates month-by-month cash flows for an MBS pool under a given PSA speed.

    The algorithm
    -------------
    For each month t:
      1. Compute SMM from the PSA CPR schedule.
      2. Scheduled interest = balance × monthly_net_rate
      3. Scheduled principal (amort) = standard mortgage formula
      4. Prepayment = SMM × (beg_balance − scheduled_principal)
      5. Total principal = scheduled + prepayment
      6. End balance = beg_balance − total_principal
    """

    def generate(
        self,
        pool:      MBSPool,
        psa_speed: float = 100.0,
    ) -> List[MBSCashFlow]:
        balance   = pool.current_balance
        n         = pool.remaining_term
        r_monthly = pool.gross_coupon / MONTHS_PER_YEAR   # use gross for amort schedule
        r_net     = pool.monthly_net_rate

        cprs     = PSAModel.cpr_schedule(psa_speed, pool.wala, n)
        smms     = np.array([PSAModel.cpr_to_smm(c) for c in cprs])

        flows: List[MBSCashFlow] = []

        for t in range(n):
            # Scheduled monthly payment (standard annuity formula)
            if r_monthly > 0:
                # Monthly payment based on GROSS coupon (this is what the borrower pays)
                pmt = balance * r_monthly / (1 - (1 + r_monthly) ** (-(n - t)))
            else:
                pmt = balance / (n - t)

            sched_interest   = balance * r_net                      # investor gets net
            sched_principal  = pmt - balance * r_monthly            # amort at gross
            sched_principal  = max(sched_principal, 0.0)

            prepayment       = smms[t] * (balance - sched_principal)
            total_principal  = sched_principal + prepayment
            total_payment    = sched_interest + total_principal
            end_balance      = max(balance - total_principal, 0.0)

            # Approximate payment date (25th of following month — standard agency delay)
            from datetime import timedelta
            pmt_date = date(
                pool.settlement_date.year  + (pool.settlement_date.month + t - 1) // 12,
                (pool.settlement_date.month + t - 1) % 12 + 1,
                25,
            )

            flows.append(MBSCashFlow(
                month               = t + 1,
                payment_date        = pmt_date,
                beg_balance         = balance,
                scheduled_interest  = sched_interest,
                scheduled_principal = sched_principal,
                prepayment          = prepayment,
                total_principal     = total_principal,
                total_payment       = total_payment,
                end_balance         = end_balance,
                smm                 = smms[t],
                cpr                 = cprs[t],
            ))

            balance = end_balance
            if balance < 1.0:   # pool fully paid down
                break

        return flows


# ──────────────────────────────────────────────────────────────────────────────
# 5.  WEIGHTED AVERAGE LIFE  (WAL)
# ──────────────────────────────────────────────────────────────────────────────

def weighted_average_life(flows: List[MBSCashFlow], settlement: date) -> float:
    """
    WAL = Σ (principal_t × t) / Σ (principal_t)
    where t is years from settlement.

    WAL is the primary duration proxy for MBS used in regulatory bucketing
    (SEC Rule 4210 requires WAL-based bucket assignment for MBS, not stated maturity).
    """
    total_principal = sum(f.total_principal for f in flows)
    if total_principal == 0:
        return 0.0

    weighted_sum = sum(
        f.total_principal * (f.payment_date - settlement).days / 365.25
        for f in flows
    )
    return weighted_sum / total_principal


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MBS PRICER  (price from OAS, risk measures via bump)
# ──────────────────────────────────────────────────────────────────────────────

class MBSPricer:
    """
    Prices MBS by discounting projected cash flows at the zero curve + OAS.

    Key difference from UST pricing
    --------------------------------
    MBS cash flows are NOT fixed — they depend on the prepayment assumption
    (PSA speed).  The OAS (Option-Adjusted Spread) is the constant spread
    over the zero curve that makes the PV of projected cash flows equal the
    market price.  It is "option-adjusted" because a full OAS model would
    simulate rate paths and average across them; here we use the simplified
    single-path version (equivalent to a Z-spread on MBS — common in
    practice for vanilla agency pass-throughs).

    Risk measures
    -------------
    All risk measures (duration, DV01, convexity) are computed by bumping
    the DISCOUNT CURVE by ±1bp and re-pricing, holding the PSA speed constant.
    This is the "OAD" (Option-Adjusted Duration) methodology.
    """

    def __init__(self, zero_curve_tenors: np.ndarray, zero_curve_rates: np.ndarray) -> None:
        self.zc_tenors = zero_curve_tenors
        self.zc_rates  = zero_curve_rates

    def _discount_rate(self, t_years: float, spread_bps: float = 0.0) -> float:
        r = float(np.interp(t_years, self.zc_tenors, self.zc_rates))
        return r + spread_bps / 10_000

    def price_from_oas(
        self,
        pool:      MBSPool,
        oas_bps:   float,
        psa_speed: float = 100.0,
    ) -> float:
        """
        Returns price per $100 current face.
        Discounts all projected cash flows at zero_curve(t) + OAS.
        """
        gen    = MBSCashFlowGenerator()
        flows  = gen.generate(pool, psa_speed)
        settle = pool.settlement_date
        pv     = 0.0

        for f in flows:
            t_years = (f.payment_date - settle).days / 365.25
            r       = self._discount_rate(t_years, oas_bps)
            df      = math.exp(-r * t_years)
            pv      += f.total_payment * df

        return pv * 100.0 / pool.current_balance

    def oas_from_price(
        self,
        pool:       MBSPool,
        price_100:  float,
        psa_speed:  float = 100.0,
    ) -> float:
        """Solves for OAS (bps) given market price per $100 current face."""
        def diff(oas):
            return self.price_from_oas(pool, oas, psa_speed) - price_100

        try:
            oas = brentq(diff, -500.0, 2000.0, xtol=1e-6, maxiter=300)
        except ValueError:
            log.warning("OAS solver failed for pool %s. Returning NaN.", pool.pool_number)
            return float("nan")
        return oas

    def full_analytics(
        self,
        pool:       MBSPool,
        price_100:  float,
        psa_speed:  float  = 100.0,
        bump_bps:   float  = 1.0,
    ) -> MBSAnalytics:
        """
        Full risk analytics for an MBS position.  DV01 and duration computed
        via parallel shift of the discount curve (OAD methodology).
        """
        gen   = MBSCashFlowGenerator()
        flows = gen.generate(pool, psa_speed)
        cf_df = pd.DataFrame([f.__dict__ for f in flows])

        oas      = self.oas_from_price(pool, price_100, psa_speed)
        wal      = weighted_average_life(flows, pool.settlement_date)

        # ── YTM (cash flow yield — single discount rate) ────────────────
        settle = pool.settlement_date
        total_cf = [(f.payment_date, f.total_payment) for f in flows]

        def ytm_diff(y):
            pv = sum(
                cf * math.exp(-y * (d - settle).days / 365.25)
                for d, cf in total_cf
            )
            return pv * 100.0 / pool.current_balance - price_100

        try:
            ytm = brentq(ytm_diff, 0.0001, 0.9999, xtol=1e-8)
        except ValueError:
            ytm = float("nan")

        # ── Duration & Convexity (bump-and-reprice) ─────────────────────
        bump = bump_bps / 10_000
        # Shift zero curve up/down
        rates_up = self.zc_rates + bump / 2
        rates_dn = self.zc_rates - bump / 2

        pricer_up = MBSPricer(self.zc_tenors, rates_up)
        pricer_dn = MBSPricer(self.zc_tenors, rates_dn)

        px_up = pricer_up.price_from_oas(pool, oas, psa_speed)
        px_dn = pricer_dn.price_from_oas(pool, oas, psa_speed)

        mod_dur  = -(px_up - px_dn) / (2 * bump * price_100)
        convexity = (px_up + px_dn - 2 * price_100) / (bump ** 2 * price_100)
        dv01_100  = (px_dn - px_up) / 2   # $ per $100 face per 1bp
        dv01      = dv01_100 / 100          # $ per $1 face per 1bp

        log.info(
            "MBS %s  PSA=%.0f  OAS=%.1fbps  WAL=%.2fyr  ModDur=%.3f  DV01=$%.4f",
            pool.pool_number, psa_speed, oas, wal, mod_dur, dv01
        )

        return MBSAnalytics(
            pool_number       = pool.pool_number,
            settlement_date   = settle,
            psa_speed         = psa_speed,
            price_100         = price_100,
            oas_bps           = oas,
            wal_years         = wal,
            modified_duration = mod_dur,
            dv01              = dv01,
            convexity         = convexity,
            yield_to_maturity = ytm,
            cash_flows        = cf_df,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PSA SENSITIVITY ANALYSIS  (required for model validation reports)
# ──────────────────────────────────────────────────────────────────────────────

class PSASensitivityAnalyzer:
    """
    Runs the pricer across a range of PSA speeds and reports how price,
    WAL, and duration change.  This is what the SEC expects to see in a
    model documentation package for MBS margining.

    Speed range: typically 50 PSA to 600 PSA for agency collateral.
    """

    def __init__(self, pricer: MBSPricer) -> None:
        self.pricer = pricer

    def sensitivity_table(
        self,
        pool:        MBSPool,
        price_100:   float,
        psa_speeds:  Optional[List[float]] = None,
    ) -> pd.DataFrame:
        psa_speeds = psa_speeds or [50, 100, 150, 200, 250, 300, 400, 500, 600]
        rows = []
        for speed in psa_speeds:
            a = self.pricer.full_analytics(pool, price_100, psa_speed=speed)
            rows.append({
                "psa_speed":       speed,
                "price_100":       price_100,
                "oas_bps":         a.oas_bps,
                "wal_years":       a.wal_years,
                "modified_duration": a.modified_duration,
                "dv01_per_1mm":    a.dv01 * 1_000_000,
                "convexity":       a.convexity,
            })
        df = pd.DataFrame(rows)
        log.info("PSA sensitivity table computed for pool %s.", pool.pool_number)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 8.  AGENCY MBS PORTFOLIO AGGREGATOR
# ──────────────────────────────────────────────────────────────────────────────

class MBSPortfolioRisk:
    """
    Aggregates risk measures across a portfolio of MBS pools.
    WAL-based bucket assignment per SEC Rule 4210 for MBS margining.
    """

    def __init__(self, pricer: MBSPricer) -> None:
        self.pricer = pricer

    def analyze_portfolio(
        self,
        positions: List[Tuple[MBSPool, float, float]],   # (pool, face_amt, price_100)
        psa_speed: float = 100.0,
    ) -> pd.DataFrame:
        rows = []
        for pool, face_amt, price_100 in positions:
            a     = self.pricer.full_analytics(pool, price_100, psa_speed)
            scale = face_amt / pool.current_balance

            rows.append({
                "pool_number":      pool.pool_number,
                "agency":           pool.agency.value,
                "current_balance":  pool.current_balance,
                "face_amount":      face_amt,
                "price_100":        price_100,
                "market_value":     face_amt * price_100 / 100,
                "psa_speed":        psa_speed,
                "oas_bps":          a.oas_bps,
                "wal_years":        a.wal_years,         # ← use for bucket, not maturity
                "modified_duration":a.modified_duration,
                "dv01_total":       a.dv01 * face_amt,
                "convexity":        a.convexity,
                "ytm":              a.yield_to_maturity,
                "wac":              pool.gross_coupon,
                "wam_months":       pool.wam,
            })

        df = pd.DataFrame(rows)

        # WAL-based bucket assignment (SEC 4210 requirement for MBS)
        def wal_bucket(wal):
            if wal <  2.0: return "0-2yr"
            if wal <  5.0: return "2-5yr"
            if wal < 10.0: return "5-10yr"
            if wal < 20.0: return "10-20yr"
            return "20-30yr"

        df["margin_bucket"] = df["wal_years"].apply(wal_bucket)

        log.info(
            "MBS portfolio: %d pools, total MV=$%,.0f, total DV01=$%,.0f",
            len(df), df["market_value"].sum(), df["dv01_total"].sum()
        )
        return df


# ──────────────────────────────────────────────────────────────────────────────
# 9.  DEMO
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    from tabulate import tabulate

    settle = date(2024, 7, 1)

    # Illustrative zero curve (mid-2024)
    zc_tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30])
    zc_rates  = np.array([0.0528, 0.0526, 0.0518, 0.0488, 0.0472, 0.0447,
                          0.0437, 0.0427, 0.0432, 0.0440, 0.0437, 0.0433])
    pricer = MBSPricer(zc_tenors, zc_rates)

    pools = [
        MBSPool(
            pool_number="MA3456", agency=AgencyType.FNMA, collateral_type=CollateralType.FIXED_RATE,
            original_balance=50_000_000, current_factor=0.85,
            gross_coupon=0.065, net_coupon=0.0625, wam=320, wala=40,
            issue_date=date(2021, 3, 1), settlement_date=settle,
        ),
        MBSPool(
            pool_number="G2SF30-887654", agency=AgencyType.GNMA, collateral_type=CollateralType.FIXED_RATE,
            original_balance=75_000_000, current_factor=0.92,
            gross_coupon=0.0700, net_coupon=0.0675, wam=340, wala=20,
            issue_date=date(2022, 11, 1), settlement_date=settle,
        ),
        MBSPool(
            pool_number="FG-A99012", agency=AgencyType.FHLMC, collateral_type=CollateralType.FIXED_RATE,
            original_balance=30_000_000, current_factor=0.60,
            gross_coupon=0.0375, net_coupon=0.0350, wam=200, wala=160,
            issue_date=date(2010, 7, 1), settlement_date=settle,
        ),
    ]
    prices = [101.25, 103.50, 94.75]

    portfolio = MBSPortfolioRisk(pricer)
    df = portfolio.analyze_portfolio(
        [(p, p.current_balance, px) for p, px in zip(pools, prices)],
        psa_speed=150.0,
    )

    print("\n" + "═" * 115)
    print("  CedarDev Capital — AGENCY MBS RISK REPORT  (150% PSA)")
    print("═" * 115)
    cols = ["pool_number","agency","market_value","oas_bps","wal_years",
            "modified_duration","dv01_total","convexity","margin_bucket"]
    print(tabulate(df[cols], headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))

    # PSA sensitivity for the FNMA pool
    print("\n  PSA SENSITIVITY — Pool MA3456 (FNMA 6.25% coupon)")
    print("─" * 80)
    sens = PSASensitivityAnalyzer(pricer).sensitivity_table(pools[0], prices[0])
    print(tabulate(sens, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
