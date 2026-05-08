"""
test_margin_engine.py
=====================
pytest suite for margin_engine.py
Run: pytest test_margin_engine.py -v --tb=short
"""

import math
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from margin_engine import (
    BUCKET_BOUNDS,
    BUCKET_ORDER,
    CORR_DF,
    PORTFOLIO_IDS,
    REG_T_MARGIN_TABLE,
    Bucket,
    BucketCategorizer,
    CrossMarginEngine,
    IntraBucketNetter,
    MarginPipeline,
    MarginValidator,
    SnowflakeExtractor,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def as_of_date():
    return date(2024, 6, 30)


@pytest.fixture
def categorizer(as_of_date):
    return BucketCategorizer(as_of=as_of_date)


@pytest.fixture
def sample_positions(as_of_date):
    """Minimal DataFrame covering every bucket, both sides."""
    rows = [
        # (maturity offset in years, side, dv01, mv, portfolio)
        (0.5,  "LONG",  100.0,  1_000_000, "MCM-RATES-001"),   # 0-2
        (1.5,  "SHORT", -80.0,    900_000, "MCM-RATES-001"),   # 0-2
        (3.0,  "LONG",  500.0,  5_000_000, "MCM-RATES-001"),   # 2-5
        (4.5,  "SHORT", -300.0, 3_000_000, "MCM-RATES-001"),   # 2-5
        (7.0,  "LONG",  900.0,  9_000_000, "MCM-RATES-001"),   # 5-10
        (7.0,  "SHORT", -400.0, 4_000_000, "MCM-RATES-001"),   # 5-10
        (15.0, "LONG",  700.0,  7_000_000, "MCM-RATES-001"),   # 10-20
        (25.0, "SHORT", -200.0, 2_000_000, "MCM-RATES-001"),   # 20-30
        (32.0, "LONG",  1500.0,15_000_000, "MCM-RATES-001"),   # 30+
    ]
    records = []
    for i, (yrs, side, dv01, mv, portf) in enumerate(rows):
        records.append({
            "position_id":       f"TST-{i:04d}",
            "portfolio_id":      portf,
            "cusip":             f"TESTCUSP{i:03d}",
            "instrument_type":   "UST",
            "maturity_date":     as_of_date + timedelta(days=int(yrs * 365.25)),
            "coupon_rate":       0.030,
            "face_value":        mv,
            "market_value_usd":  mv,
            "dv01_usd":          dv01,
            "side":              side,
            "settlement_date":   as_of_date + timedelta(days=2),
            "trader_id":         "T001_TEST",
            "book_id":           "BOOK_TEST",
            "asset_class":       "RATES",
            "modified_duration": abs(dv01) / mv * 10_000,
            "convexity":         0.5,
            "oas_spread":        None,
            "last_updated_utc":  "2024-06-30T18:00:00Z",
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — BucketCategorizer
# ──────────────────────────────────────────────────────────────────────────────

class TestBucketCategorizer:

    def test_bucket_boundaries_exact(self, as_of_date):
        """Boundary values map to the UPPER bucket ([lo, hi))."""
        cat = BucketCategorizer(as_of=as_of_date)
        # Exactly 2 years → B_2_5 (not B_0_2)
        mat = as_of_date + timedelta(days=int(2.0 * 365.25))
        assert cat.assign_bucket(mat) == Bucket.B_2_5

    def test_all_buckets_covered(self, categorizer):
        """Every bucket enum value must be reachable via assign_bucket."""
        as_of = categorizer.as_of
        midpoints_yrs = [1.0, 3.5, 7.5, 15.0, 25.0, 35.0]
        buckets_hit = {
            categorizer.assign_bucket(as_of + timedelta(days=int(y * 365.25)))
            for y in midpoints_yrs
        }
        assert buckets_hit == set(BUCKET_ORDER)

    def test_categorize_adds_columns(self, categorizer, sample_positions):
        df = categorizer.categorize(sample_positions)
        assert "ytm_years" in df.columns
        assert "bucket"    in df.columns
        assert "bucket_label" in df.columns
        assert df["ytm_years"].isna().sum() == 0

    def test_past_maturity_gracefully_handled(self, categorizer, as_of_date):
        mat = as_of_date - timedelta(days=30)   # already past
        bucket = categorizer.assign_bucket(mat)
        assert bucket == Bucket.B_0_2


# ──────────────────────────────────────────────────────────────────────────────
# Tests — IntraBucketNetter
# ──────────────────────────────────────────────────────────────────────────────

class TestIntraBucketNetter:

    def test_net_dv01_sum(self, categorizer, sample_positions):
        df = categorizer.categorize(sample_positions)
        netter = IntraBucketNetter()
        netting_df, _ = netter.net(df)
        row = netting_df[
            (netting_df["portfolio_id"] == "MCM-RATES-001") &
            (netting_df["bucket_label"] == Bucket.B_0_2.value)
        ].iloc[0]
        assert math.isclose(row["long_dv01"],   100.0, rel_tol=1e-9)
        assert math.isclose(row["short_dv01"],  -80.0, rel_tol=1e-9)
        assert math.isclose(row["net_dv01"],     20.0, rel_tol=1e-9)

    def test_no_nulls_in_output(self, categorizer, sample_positions):
        df = categorizer.categorize(sample_positions)
        netting_df, _ = IntraBucketNetter().net(df)
        assert netting_df[["long_dv01", "short_dv01", "net_dv01"]].isna().sum().sum() == 0

    def test_gross_mv_always_positive(self, categorizer, sample_positions):
        df = categorizer.categorize(sample_positions)
        netting_df, _ = IntraBucketNetter().net(df)
        assert (netting_df["gross_mv"] >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Correlation Matrix
# ──────────────────────────────────────────────────────────────────────────────

class TestCorrelationMatrix:

    def test_diagonal_is_one(self):
        assert np.allclose(np.diag(CORR_DF.values), 1.0)

    def test_symmetric(self):
        assert np.allclose(CORR_DF.values, CORR_DF.values.T)

    def test_all_entries_in_0_1(self):
        vals = CORR_DF.values
        assert (vals >= 0).all() and (vals <= 1).all()

    def test_adjacent_buckets_higher_than_distant(self):
        """5-10yr ↔ 10-20yr corr must exceed 5-10yr ↔ 30yr+ corr."""
        corr_adj  = CORR_DF.loc["5-10yr", "10-20yr"]
        corr_dist = CORR_DF.loc["5-10yr", "30yr+"]
        assert corr_adj > corr_dist


# ──────────────────────────────────────────────────────────────────────────────
# Tests — CrossMarginEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossMarginEngine:

    def _run_pipeline(self, sample_positions, as_of_date):
        cat = BucketCategorizer(as_of=as_of_date)
        df  = cat.categorize(sample_positions)
        net, _ = IntraBucketNetter().net(df)
        engine  = CrossMarginEngine()
        results_df, detailed = engine.compute(net)
        return results_df, detailed

    def test_final_margin_non_negative(self, sample_positions, as_of_date):
        results_df, _ = self._run_pipeline(sample_positions, as_of_date)
        assert (results_df["final_margin"] >= 0).all()

    def test_credit_does_not_exceed_gross(self, sample_positions, as_of_date):
        results_df, _ = self._run_pipeline(sample_positions, as_of_date)
        assert (results_df["cross_margin_credit"] <= results_df["gross_margin"] + 1e-2).all()

    def test_no_credit_for_same_direction_buckets(self, as_of_date):
        """
        If all positions are LONG, there should be zero cross-margin credit.
        """
        rows = [
            (3.0, "LONG", 500.0, 5_000_000, "MCM-RATES-001"),
            (7.0, "LONG", 900.0, 9_000_000, "MCM-RATES-001"),
        ]
        records = []
        for i, (yrs, side, dv01, mv, portf) in enumerate(rows):
            records.append({
                "position_id":       f"SD-{i:04d}",
                "portfolio_id":      portf,
                "cusip":             f"SDCUSP{i:03d}",
                "instrument_type":   "UST",
                "maturity_date":     as_of_date + timedelta(days=int(yrs * 365.25)),
                "coupon_rate":       0.030,
                "face_value":        mv,
                "market_value_usd":  mv,
                "dv01_usd":          dv01,
                "side":              side,
                "settlement_date":   as_of_date + timedelta(days=2),
                "trader_id":         "T001",
                "book_id":           "BOOK",
                "asset_class":       "RATES",
                "modified_duration": 5.0,
                "convexity":         0.5,
                "oas_spread":        None,
                "last_updated_utc":  "2024-06-30T18:00:00Z",
            })
        df = pd.DataFrame(records)
        cat = BucketCategorizer(as_of=as_of_date)
        df  = cat.categorize(df)
        net, _ = IntraBucketNetter().net(df)
        results_df, _ = CrossMarginEngine().compute(net)
        total_credit = results_df["cross_margin_credit"].sum()
        assert total_credit == pytest.approx(0.0, abs=1e-2)

    def test_audit_notes_present(self, sample_positions, as_of_date):
        _, detailed = self._run_pipeline(sample_positions, as_of_date)
        for r in detailed:
            assert len(r.audit_notes) > 0, f"No audit notes for {r.bucket}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests — MarginValidator
# ──────────────────────────────────────────────────────────────────────────────

class TestMarginValidator:

    def _build_valid_results_df(self, as_of_date, sample_positions):
        cat = BucketCategorizer(as_of=as_of_date)
        df  = cat.categorize(sample_positions)
        net, _ = IntraBucketNetter().net(df)
        res_df, detailed = CrossMarginEngine().compute(net)
        return res_df, detailed

    def test_valid_output_passes(self, as_of_date, sample_positions):
        res_df, detailed = self._build_valid_results_df(as_of_date, sample_positions)
        v = MarginValidator().validate(res_df, detailed)
        assert v.passed

    def test_negative_margin_triggers_failure(self, as_of_date, sample_positions):
        res_df, detailed = self._build_valid_results_df(as_of_date, sample_positions)
        # Corrupt one row
        res_df.loc[0, "final_margin"] = -500.0
        v = MarginValidator().validate(res_df, detailed)
        assert not v.passed
        assert any("Negative" in m for m in v.mismatches)

    def test_wrong_rate_triggers_failure(self, as_of_date, sample_positions):
        res_df, detailed = self._build_valid_results_df(as_of_date, sample_positions)
        res_df.loc[0, "margin_rate"] = 0.9999   # obviously wrong
        v = MarginValidator().validate(res_df, detailed)
        assert not v.passed

    def test_checksum_ok(self, as_of_date, sample_positions):
        res_df, detailed = self._build_valid_results_df(as_of_date, sample_positions)
        v = MarginValidator().validate(res_df, detailed)
        assert v.checksum_ok


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Full Pipeline (integration)
# ──────────────────────────────────────────────────────────────────────────────

class TestMarginPipeline:

    def test_synthetic_pipeline_end_to_end(self, tmp_path):
        pipeline = MarginPipeline(use_synthetic=True, output_dir=tmp_path)
        results_df, validation = pipeline.run()
        assert validation.passed
        assert len(results_df) > 0
        assert "final_margin" in results_df.columns

    def test_output_csvs_written(self, tmp_path):
        pipeline = MarginPipeline(use_synthetic=True, output_dir=tmp_path)
        pipeline.run()
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) >= 2   # netting + margins

    def test_all_portfolios_in_output(self, tmp_path):
        pipeline = MarginPipeline(use_synthetic=True, output_dir=tmp_path)
        results_df, _ = pipeline.run()
        for portf in PORTFOLIO_IDS:
            assert portf in results_df["portfolio_id"].values, (
                f"Portfolio {portf} missing from results."
            )
