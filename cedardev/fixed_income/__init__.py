"""
cedardev.fixed_income
=====================
Fixed Income Risk Analytics — CedarDev Capital Management LLC

Public API
----------
Pricers
    from cedardev.fixed_income import USTBondPricer, USTBond, USTAnalytics
    from cedardev.fixed_income import MBSPricer, MBSPool, MBSAnalytics
    from cedardev.fixed_income import CTDEngine, FuturesPosition, FuturesContract

Risk
    from cedardev.fixed_income import USTPortfolioRisk
    from cedardev.fixed_income import MBSPortfolioRisk, PSASensitivityAnalyzer
    from cedardev.fixed_income import HistoricalSimVaR, ParametricVaR

Curves
    from cedardev.fixed_income import ZeroCurve

Margin
    from cedardev.fixed_income import MarginPipeline, BucketCategorizer
    from cedardev.fixed_income import IntraBucketNetter, CrossMarginEngine
    from cedardev.fixed_income import MarginValidator

Validation
    from cedardev.fixed_income import ModelValidator, KupiecTest
    from cedardev.fixed_income import ChristoffersenTest, BaselTrafficLightAssigner
"""

# ── Pricers ───────────────────────────────────────────────────────────────────
from cedardev.fixed_income.ust_pricer import (
    USTBond,
    USTAnalytics,
    USTBondPricer,
    USTPortfolioRisk,
    ZeroCurve,
    ActActICMA,
    DayCount,
    Frequency,
    generate_cash_flows,
)

from cedardev.fixed_income.mbs_analytics import (
    MBSPool,
    MBSAnalytics,
    MBSCashFlow,
    MBSPricer,
    MBSPortfolioRisk,
    MBSCashFlowGenerator,
    PSAModel,
    PSASensitivityAnalyzer,
    AgencyType,
    CollateralType,
    weighted_average_life,
)

from cedardev.fixed_income.treasury_futures import (
    FuturesContract,
    FuturesPosition,
    CTDResult,
    CTDEngine,
    CMEContractSpec,
    CME_SPECS,
    ConversionFactorCalculator,
    theoretical_futures_price,
    futures_portfolio_dv01,
)

# ── VaR ───────────────────────────────────────────────────────────────────────
from cedardev.fixed_income.var_engine import (
    PortfolioPosition,
    VaRResult,
    HistoricalSimVaR,
    ParametricVaR,
    KeyRateDV01Mapper,
    VaRReporter,
    KEY_RATE_TENORS,
    load_yield_curve_history,
)

# ── Margin Engine ─────────────────────────────────────────────────────────────
from cedardev.fixed_income.margin_engine import (
    Bucket,
    BucketCategorizer,
    IntraBucketNetter,
    CrossMarginEngine,
    MarginValidator,
    MarginPipeline,
    MarginReporter,
    MarginResult,
    BucketSummary,
    REG_T_MARGIN_TABLE,
    CORR_DF,
    BUCKET_ORDER,
)

# ── Model Validation ──────────────────────────────────────────────────────────
from cedardev.fixed_income.model_validation import (
    BacktestWindow,
    KupiecResult,
    ChristoffersenResult,
    BaselTrafficLight,
    ValidationReport,
    KupiecTest,
    ChristoffersenTest,
    BaselTrafficLightAssigner,
    PLATest,
    ModelValidator,
)

__version__ = "1.0.0"
__author__  = "CedarDev Capital Management LLC"
__all__ = [
    # UST Pricer
    "USTBond", "USTAnalytics", "USTBondPricer", "USTPortfolioRisk",
    "ZeroCurve", "ActActICMA", "DayCount", "Frequency", "generate_cash_flows",
    # MBS
    "MBSPool", "MBSAnalytics", "MBSCashFlow", "MBSPricer", "MBSPortfolioRisk",
    "MBSCashFlowGenerator", "PSAModel", "PSASensitivityAnalyzer",
    "AgencyType", "CollateralType", "weighted_average_life",
    # Futures
    "FuturesContract", "FuturesPosition", "CTDResult", "CTDEngine",
    "CMEContractSpec", "CME_SPECS", "ConversionFactorCalculator",
    "theoretical_futures_price", "futures_portfolio_dv01",
    # VaR
    "PortfolioPosition", "VaRResult", "HistoricalSimVaR", "ParametricVaR",
    "KeyRateDV01Mapper", "VaRReporter", "KEY_RATE_TENORS", "load_yield_curve_history",
    # Margin
    "Bucket", "BucketCategorizer", "IntraBucketNetter", "CrossMarginEngine",
    "MarginValidator", "MarginPipeline", "MarginReporter", "MarginResult",
    "BucketSummary", "REG_T_MARGIN_TABLE", "CORR_DF", "BUCKET_ORDER",
    # Validation
    "BacktestWindow", "KupiecResult", "ChristoffersenResult", "BaselTrafficLight",
    "ValidationReport", "KupiecTest", "ChristoffersenTest",
    "BaselTrafficLightAssigner", "PLATest", "ModelValidator",
]
