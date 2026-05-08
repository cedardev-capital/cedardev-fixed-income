"""
treasury_futures.py
===================
US Treasury Futures Analytics
Firm: CedarDev Capital Management LLC  |  Desk: Market Risk Infrastructure
Version: 1.6.0  |  Python 3.11+

Covers
------
  - CME Treasury futures contracts (ZT, ZF, ZN, ZB, UB)
  - Conversion factor (CF) calculation per CME methodology
  - Cheapest-to-Deliver (CTD) bond identification
  - Theoretical futures price (cost-of-carry model)
  - Implied repo rate (breakeven carry)
  - Gross basis and net basis
  - DV01 of a futures position (via CTD DV01 and CF)
  - Invoice price calculation (for delivery)

Regulatory Context
------------------
  Treasury futures positions must be margined separately from cash bonds.
  The CTD-equivalent DV01 feeds into the cross-margining calculation in
  margin_engine.py to determine the net risk exposure of a cash-futures
  hedged position.

CME Contract Specs (as of 2024)
--------------------------------
  2yr  (ZT): 6% notional coupon, $200k face, 1yr9m–2yr maturity deliverable
  5yr  (ZF): 6% notional coupon, $100k face, 4yr2m–5yr3m deliverable
  10yr (ZN): 6% notional coupon, $100k face, 6yr6m–10yr deliverable
  30yr (ZB): 6% notional coupon, $100k face, 15yr–25yr deliverable
  Ultra (UB): 6% notional coupon, $100k face, 25yr+ deliverable

References
----------
  - CME Group, "U.S. Treasury Futures Delivery" (CME Publication)
  - Burghardt & Hoskins, "The Treasury Bond Basis" (McGraw-Hill)
  - Fabozzi, "Fixed Income Mathematics", Ch. 12
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

from cedardev.fixed_income.ust_pricer import USTBond, USTBondPricer, ActActICMA, generate_cash_flows, ZeroCurve

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONTRACT SPECIFICATIONS
# ──────────────────────────────────────────────────────────────────────────────

class FuturesContract(str, Enum):
    ZT  = "2yr Note Futures"
    ZF  = "5yr Note Futures"
    ZN  = "10yr Note Futures"
    ZB  = "30yr Bond Futures"
    UB  = "Ultra Bond Futures"


@dataclass(frozen=True)
class CMEContractSpec:
    """Static CME contract specification."""
    symbol:                FuturesContract
    notional_coupon:       float      # 6% for all Treasury futures
    face_value:            float      # contract face ($)
    min_maturity_years:    float      # deliverable basket lower bound
    max_maturity_years:    float      # deliverable basket upper bound
    tick_size:             float      # price in 32nds per tick
    tick_value:            float      # $ per tick
    delivery_months:       List[str]  # March, June, September, December


CME_SPECS: Dict[FuturesContract, CMEContractSpec] = {
    FuturesContract.ZT: CMEContractSpec(
        FuturesContract.ZT, 0.06, 200_000, 1.75, 2.00,
        tick_size=1/128, tick_value=15.625,
        delivery_months=["Mar","Jun","Sep","Dec"],
    ),
    FuturesContract.ZF: CMEContractSpec(
        FuturesContract.ZF, 0.06, 100_000, 4.167, 5.25,
        tick_size=1/128, tick_value=7.8125,
        delivery_months=["Mar","Jun","Sep","Dec"],
    ),
    FuturesContract.ZN: CMEContractSpec(
        FuturesContract.ZN, 0.06, 100_000, 6.5, 10.0,
        tick_size=1/64, tick_value=15.625,
        delivery_months=["Mar","Jun","Sep","Dec"],
    ),
    FuturesContract.ZB: CMEContractSpec(
        FuturesContract.ZB, 0.06, 100_000, 15.0, 25.0,
        tick_size=1/32, tick_value=31.25,
        delivery_months=["Mar","Jun","Sep","Dec"],
    ),
    FuturesContract.UB: CMEContractSpec(
        FuturesContract.UB, 0.06, 100_000, 25.0, 999.0,
        tick_size=1/32, tick_value=31.25,
        delivery_months=["Mar","Jun","Sep","Dec"],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FuturesPosition:
    """A live Treasury futures position."""
    contract:          FuturesContract
    expiry_date:       date             # last trading / delivery date
    futures_price:     float            # quoted futures price (per $100 face)
    num_contracts:     int              # signed: positive = long, negative = short
    delivery_date:     date             # first delivery date of the front contract


@dataclass
class CTDResult:
    """Output of CTD identification for a given futures contract."""
    bond:              USTBond
    conversion_factor: float
    gross_basis:       float    # cash price − futures × CF  (32nds)
    net_basis:         float    # gross basis − carry         (32nds)
    implied_repo:      float    # breakeven financing rate (annual)
    invoice_price:     float    # what short receives on delivery
    dv01_futures:      float    # DV01 of 1 futures contract ($ per bp)
    ctd_clean_price:   float
    delivery_profit:   float    # short's P&L if delivered today


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CONVERSION FACTOR  (CME Methodology)
# ──────────────────────────────────────────────────────────────────────────────

class ConversionFactorCalculator:
    """
    Computes the CME conversion factor for a deliverable Treasury bond.

    Definition
    ----------
    The conversion factor is the price of the deliverable bond (per $1 face)
    if it were priced to yield the notional coupon rate of the futures contract
    (6%), with cash flows rounded to the nearest quarter.

    CME Algorithm
    -------------
    1. Determine the number of full and fractional quarters from the first
       day of the delivery month to the maturity date.
    2. Round DOWN to the nearest quarter (N full quarters + z months 0 or 3).
    3. Price the bond at that rounded maturity using the 6% notional yield.
    4. Subtract accrued interest as of the first day of the delivery month.

    This is NOT a YTM calculation — it's a fixed-yield pricing.
    """

    NOTIONAL_YIELD = 0.06    # 6% for all CME Treasury futures

    def calculate(
        self,
        bond:              USTBond,
        delivery_month:    date,    # first day of the delivery month
    ) -> float:
        """
        Returns the conversion factor (dimensionless, typically near 1.0).
        """
        # ── Step 1: months from delivery month to maturity ────────────────
        months_to_mat = (
            (bond.maturity_date.year  - delivery_month.year)  * 12 +
            (bond.maturity_date.month - delivery_month.month)
        )

        # ── Step 2: round DOWN to nearest 3-month quarter ─────────────────
        full_quarters   = months_to_mat // 3
        remainder_months = months_to_mat % 3
        # z = 0 if remainder < 7 weeks, else 3  (CME approximation: 0 or 3)
        z = 3 if remainder_months >= 2 else 0
        n_periods = full_quarters * 1   # in quarters (each = 3 months)
        # Effective maturity in semi-annual periods
        effective_months = full_quarters * 3 + z
        effective_periods = effective_months / 6    # semi-annual

        # ── Step 3: price at 6% semi-annual yield ─────────────────────────
        r_semi = self.NOTIONAL_YIELD / 2
        c_semi = bond.coupon_rate   / 2   # semi-annual coupon per $1 face

        n = effective_periods
        if n <= 0:
            return 1.0

        if z == 0:
            # Coupon date coincides with delivery month → no stub
            cf = (
                c_semi * (1 - (1 + r_semi) ** (-n)) / r_semi
                + 1 / (1 + r_semi) ** n
            )
        else:
            # z == 3: next coupon is 3 months away; half-period discount
            cf_full = (
                c_semi * (1 - (1 + r_semi) ** (-n)) / r_semi
                + 1 / (1 + r_semi) ** n
            )
            cf = cf_full / (1 + r_semi) ** 0.5  # half-period discounting

        # ── Step 4: subtract accrued as of delivery month ─────────────────
        ai_frac = (z / 6) * c_semi   # fraction of semi-annual coupon accrued
        cf -= ai_frac

        return round(cf, 4)   # CME rounds to 4 decimal places


# ──────────────────────────────────────────────────────────────────────────────
# 4.  THEORETICAL FUTURES PRICE  (cost-of-carry model)
# ──────────────────────────────────────────────────────────────────────────────

def theoretical_futures_price(
    bond:           USTBond,
    cf:             float,
    clean_price:    float,
    settlement:     date,
    delivery_date:  date,
    repo_rate:      float,    # financing rate (annual, decimal)
) -> float:
    """
    Theoretical futures price:
        F* = (Dirty_Price − PV_Coupons) × (1 + repo × T) / CF

    Where:
      - Dirty_Price   = clean_price + accrued_interest
      - PV_Coupons    = PV of coupons paid between settlement and delivery
      - T             = (delivery_date − settlement) / 360  (repo convention ACT/360)
      - CF            = conversion factor

    This is the "fair value" futures price. If market price ≠ F*,
    there is a net basis (arbitrage or delivery option value).
    """
    pricer = USTBondPricer()
    dirty, clean, ai = pricer.price_from_yield(
        bond,
        pricer.yield_from_price(bond, clean_price, settlement),
        settlement,
    )
    dirty_price = dirty   # per $100 face

    # ── PV of intermediate coupons (coupon dates between settle and delivery)
    flows = generate_cash_flows(bond, settlement)
    T_repo  = (delivery_date - settlement).days / 360.0
    T_settle = settlement

    pv_coupons = 0.0
    for pmt_date, cf_amt in flows:
        if settlement < pmt_date <= delivery_date:
            # Only coupon, not principal (for intermediate payments)
            coupon_only = bond.semi_annual_coupon * 100.0 / bond.face_value
            t = (delivery_date - pmt_date).days / 360.0
            pv_coupons += coupon_only * (1 + repo_rate * t)  # FV at delivery

    fv_bond = (dirty_price - pv_coupons / (1 + repo_rate * T_repo)) * (1 + repo_rate * T_repo)
    # Simplification: adjust_dirty = dirty_price × (1 + repo × T) − FV_coupons
    adjust_dirty = dirty_price * (1 + repo_rate * T_repo) - pv_coupons
    theoretical  = adjust_dirty / cf

    return theoretical


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CTD IDENTIFICATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class CTDEngine:
    """
    Identifies the Cheapest-to-Deliver (CTD) bond for a given futures contract.

    The CTD is the bond that minimizes the short's delivery cost:
        Delivery Cost = Clean_Price − (Futures_Price × CF)
    Equivalently, it maximizes the short's invoice price relative to acquisition cost.

    For regulatory margining, the CTD's DV01 and CF are used to convert
    a futures DV01 to a cash-equivalent DV01:
        DV01_futures = DV01_CTD / CF
    """

    def __init__(self, pricer: USTBondPricer) -> None:
        self.pricer = pricer
        self.cf_calc = ConversionFactorCalculator()

    def find_ctd(
        self,
        futures:         FuturesPosition,
        deliverable_bonds: List[Tuple[USTBond, float]],  # (bond, clean_price)
        repo_rate:       float,
        settlement:      date,
    ) -> Tuple[CTDResult, pd.DataFrame]:
        """
        Scans all deliverable bonds and returns the CTD + a full comparison table.
        """
        spec           = CME_SPECS[futures.contract]
        delivery_month = date(futures.delivery_date.year, futures.delivery_date.month, 1)

        rows = []
        best_net_basis = None
        ctd_result     = None

        for bond, clean_px in deliverable_bonds:
            # Validate deliverability window
            ytm_yr = (bond.maturity_date - settlement).days / 365.25
            if not (spec.min_maturity_years <= ytm_yr <= spec.max_maturity_years):
                log.debug("Bond %s YTM=%.2fyr outside deliverable window. Skipping.", bond.cusip, ytm_yr)
                continue

            cf = self.cf_calc.calculate(bond, delivery_month)

            # Invoice price: what the short receives when delivering
            invoice_price = (futures.futures_price * cf + self._accrued_100(bond, futures.delivery_date))

            # Gross basis (in price points, i.e. per $100 face)
            gross_basis = clean_px - futures.futures_price * cf

            # Carry: income earned on the bond minus financing cost
            carry = self._carry(bond, clean_px, settlement, futures.delivery_date, repo_rate)

            # Net basis = gross basis - carry
            net_basis = gross_basis - carry

            # Implied repo: solve for repo rate that makes net_basis = 0
            implied_repo = self._implied_repo(
                bond, cf, clean_px, settlement, futures.delivery_date, futures.futures_price
            )

            # DV01 of ONE futures contract via CTD
            analytics      = self.pricer.risk_measures(
                bond,
                self.pricer.yield_from_price(bond, clean_px, settlement),
                settlement,
            )
            dv01_ctd_per_face = analytics.dv01   # $ per $1 face per bp
            dv01_futures      = (dv01_ctd_per_face * spec.face_value) / cf

            delivery_profit = invoice_price - clean_px - self._accrued_100(bond, settlement)

            row = {
                "cusip":            bond.cusip,
                "maturity_date":    bond.maturity_date,
                "coupon_rate":      bond.coupon_rate,
                "clean_price":      clean_px,
                "conv_factor":      cf,
                "invoice_price":    invoice_price,
                "gross_basis_32nds":gross_basis * 32,
                "net_basis_32nds":  net_basis * 32,
                "implied_repo_pct": implied_repo * 100,
                "dv01_futures_usd": dv01_futures,
                "is_ctd":           False,
            }
            rows.append(row)

            # CTD = bond with lowest net basis (cheapest to deliver)
            if best_net_basis is None or net_basis < best_net_basis:
                best_net_basis = net_basis
                ctd_result = CTDResult(
                    bond              = bond,
                    conversion_factor = cf,
                    gross_basis       = gross_basis * 32,
                    net_basis         = net_basis   * 32,
                    implied_repo      = implied_repo,
                    invoice_price     = invoice_price,
                    dv01_futures      = dv01_futures,
                    ctd_clean_price   = clean_px,
                    delivery_profit   = delivery_profit,
                )

        if not rows:
            raise ValueError("No deliverable bonds found in the deliverable basket.")

        df = pd.DataFrame(rows)
        if ctd_result:
            df.loc[df["cusip"] == ctd_result.bond.cusip, "is_ctd"] = True

        log.info(
            "CTD identified: %s  CF=%.4f  Implied Repo=%.4f%%  Net Basis=%.3f/32",
            ctd_result.bond.cusip if ctd_result else "N/A",
            ctd_result.conversion_factor if ctd_result else 0,
            ctd_result.implied_repo * 100 if ctd_result else 0,
            ctd_result.net_basis if ctd_result else 0,
        )
        return ctd_result, df

    def _accrued_100(self, bond: USTBond, settlement: date) -> float:
        ai, _, _ = ActActICMA.accrued_interest(bond, settlement)
        return ai * 100.0 / bond.face_value

    def _carry(
        self,
        bond:          USTBond,
        clean_px:      float,
        settlement:    date,
        delivery_date: date,
        repo_rate:     float,
    ) -> float:
        """
        Carry = Coupon Income − Financing Cost (per $100 face).
        Coupon income: accrued interest at delivery − accrued at settlement.
        Financing:     dirty price × repo rate × T (ACT/360).
        """
        ai_settle   = self._accrued_100(bond, settlement)
        ai_delivery = self._accrued_100(bond, delivery_date)
        dirty       = clean_px + ai_settle
        T           = (delivery_date - settlement).days / 360.0
        coupon_income = ai_delivery - ai_settle
        financing     = dirty * repo_rate * T
        return coupon_income - financing

    def _implied_repo(
        self,
        bond:          USTBond,
        cf:            float,
        clean_px:      float,
        settlement:    date,
        delivery_date: date,
        futures_price: float,
    ) -> float:
        """
        Implied repo: annualized return from buying bond + delivering into futures.
            IR = (Invoice − Dirty_Settle) / Dirty_Settle × (360 / T)
        """
        ai_settle   = self._accrued_100(bond, settlement)
        ai_delivery = self._accrued_100(bond, delivery_date)
        dirty_settle   = clean_px + ai_settle
        invoice_price  = futures_price * cf + ai_delivery
        T = (delivery_date - settlement).days / 360.0
        if dirty_settle == 0 or T == 0:
            return 0.0
        return (invoice_price - dirty_settle) / dirty_settle / T


# ──────────────────────────────────────────────────────────────────────────────
# 6.  FUTURES DV01 FOR MARGIN ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def futures_portfolio_dv01(
    positions:        List[Tuple[FuturesPosition, CTDResult]],
) -> pd.DataFrame:
    """
    Converts futures positions to cash-equivalent DV01 for margin aggregation.
    Each futures contract maps to its CTD DV01 / CF.

    This output is designed to be merged with the cash bond DV01 table
    in margin_engine.py for the cross-margining calculation.
    """
    rows = []
    for fut, ctd in positions:
        spec        = CME_SPECS[fut.contract]
        dv01_signed = ctd.dv01_futures * fut.num_contracts
        rows.append({
            "instrument":       fut.contract.value,
            "expiry":           fut.expiry_date,
            "num_contracts":    fut.num_contracts,
            "futures_price":    fut.futures_price,
            "ctd_cusip":        ctd.bond.cusip,
            "conv_factor":      ctd.conversion_factor,
            "dv01_per_contract":ctd.dv01_futures,
            "dv01_total":       dv01_signed,
            "ytm_years_ctd":    (ctd.bond.maturity_date - fut.delivery_date).days / 365.25,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  DEMO
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    from tabulate import tabulate

    settle     = date(2024, 7, 1)
    delivery   = date(2024, 9, 30)
    pricer     = USTBondPricer()
    ctd_engine = CTDEngine(pricer)

    # Deliverable basket for ZN (10yr Note Futures)
    deliverable_bonds = [
        USTBond("91282CDB2", date(2018, 11,15), date(2028,11,15), 0.0313),
        USTBond("91282CEX5", date(2021,  8,15), date(2031, 7,31), 0.0125),
        USTBond("912828YK0", date(2019,  5,15), date(2029, 5,15), 0.0238),
        USTBond("91282CKH0", date(2021, 11,15), date(2029,11,15), 0.0125),
        USTBond("91282CCF6", date(2019,  8,15), date(2029, 8,15), 0.0163),
    ]
    clean_prices = [88.50, 84.10, 86.40, 85.20, 87.10]

    futures = FuturesPosition(
        contract      = FuturesContract.ZN,
        expiry_date   = delivery,
        futures_price = 109.25,
        num_contracts = 100,
        delivery_date = delivery,
    )

    ctd, basket_df = ctd_engine.find_ctd(
        futures,
        list(zip(deliverable_bonds, clean_prices)),
        repo_rate  = 0.0530,
        settlement = settle,
    )

    print("\n" + "═" * 100)
    print("  CedarDev Capital — ZN FUTURES CTD ANALYSIS")
    print("═" * 100)
    print(tabulate(basket_df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
    print(f"\n  CTD Bond      : {ctd.bond.cusip}")
    print(f"  Conv Factor   : {ctd.conversion_factor:.4f}")
    print(f"  Invoice Price : ${ctd.invoice_price:.6f} per $100 face")
    print(f"  Implied Repo  : {ctd.implied_repo*100:.4f}%")
    print(f"  DV01/contract : ${ctd.dv01_futures:,.2f}")
    print(f"  DV01 (100 lots): ${ctd.dv01_futures * 100:,.0f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
