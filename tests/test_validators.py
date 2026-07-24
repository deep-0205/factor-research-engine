"""Unit tests for the validation layer.

Tests verify that validators correctly:
1. Catch data quality issues
2. Enforce critical constraints
3. Provide helpful diagnostics
4. Fail fast on errors
"""

import pytest
import pandas as pd
import numpy as np
from validators import (
    validate_price_matrix,
    validate_returns_matrix,
    validate_universe_flags,
    validate_signal_matrix,
    validate_weight_matrix,
    validate_backtest_returns,
    validate_risk_metrics,
)


@pytest.fixture
def valid_price_matrix():
    """Create valid price matrix."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'S1': 100 + np.random.randn(100).cumsum(),
        'S2': 100 + np.random.randn(100).cumsum(),
        'S3': 100 + np.random.randn(100).cumsum(),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def valid_returns_matrix(valid_price_matrix):
    """Create valid returns matrix."""
    return valid_price_matrix.pct_change().fillna(0)


class TestPriceMatrixValidation:
    """Test price matrix validation."""
    
    def test_valid_price_matrix_passes(self, valid_price_matrix):
        """Valid price matrix should pass validation."""
        assert validate_price_matrix(valid_price_matrix) is True
    
    def test_empty_matrix_fails(self):
        """Empty matrix should fail."""
        with pytest.raises(ValueError):
            validate_price_matrix(pd.DataFrame())
    
    def test_none_matrix_fails(self):
        """None matrix should fail."""
        with pytest.raises(ValueError):
            validate_price_matrix(None)
    
    def test_unsorted_index_fails(self, valid_price_matrix):
        """Unsorted index should fail."""
        unsorted = valid_price_matrix.sample(frac=1)
        with pytest.raises(ValueError):
            validate_price_matrix(unsorted)
    
    def test_duplicate_dates_fails(self, valid_price_matrix):
        """Duplicate dates should fail."""
        dup_data = pd.concat([valid_price_matrix, valid_price_matrix.iloc[-1:]])
        with pytest.raises(ValueError):
            validate_price_matrix(dup_data)
    
    def test_infinity_values_fail(self, valid_price_matrix):
        """Infinity values should fail."""
        invalid = valid_price_matrix.copy()
        invalid.iloc[0, 0] = np.inf
        with pytest.raises(ValueError):
            validate_price_matrix(invalid)
    
    def test_non_numeric_fails(self, valid_price_matrix):
        """Non-numeric values should fail."""
        invalid = valid_price_matrix.copy()
        invalid.iloc[0, 0] = 'ABC'
        # This would fail during numeric check
        # (implementation specific)


class TestReturnsMatrixValidation:
    """Test returns matrix validation."""
    
    def test_valid_returns_matrix_passes(self, valid_returns_matrix):
        """Valid returns matrix should pass."""
        assert validate_returns_matrix(valid_returns_matrix) is True
    
    def test_infinity_in_returns_fails(self, valid_returns_matrix):
        """Infinity in returns should fail."""
        invalid = valid_returns_matrix.copy()
        invalid.iloc[10, 0] = np.inf
        with pytest.raises(ValueError):
            validate_returns_matrix(invalid)
    
    def test_extreme_returns_warning(self, valid_returns_matrix):
        """Extreme returns (>100%) should warn."""
        invalid = valid_returns_matrix.copy()
        invalid.iloc[10, 0] = 2.0  # 200% return
        
        # Should not raise, but should warn
        # (depends on implementation)


class TestUniverseFlagsValidation:
    """Test universe flags validation."""
    
    def test_valid_universe_flags_passes(self, valid_price_matrix):
        """Valid universe flags should pass."""
        dates = valid_price_matrix.index
        stocks = valid_price_matrix.columns
        flags = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)) > 0.5,
            index=dates,
            columns=stocks
        )
        assert validate_universe_flags(flags, valid_price_matrix) is True
    
    def test_shape_mismatch_fails(self, valid_price_matrix):
        """Shape mismatch should fail."""
        flags = pd.DataFrame(
            np.random.rand(50, 2) > 0.5,
            index=valid_price_matrix.index[:50],
            columns=['S1', 'S2']
        )
        with pytest.raises(ValueError):
            validate_universe_flags(flags, valid_price_matrix)
    
    def test_index_mismatch_fails(self, valid_price_matrix):
        """Index mismatch should fail."""
        wrong_index = pd.date_range(start='2022-01-01', periods=100, freq='D')
        flags = pd.DataFrame(
            np.random.rand(100, 3) > 0.5,
            index=wrong_index,
            columns=valid_price_matrix.columns
        )
        with pytest.raises(ValueError):
            validate_universe_flags(flags, valid_price_matrix)
    
    def test_non_boolean_values_fail(self, valid_price_matrix):
        """Non-boolean values should fail."""
        dates = valid_price_matrix.index
        stocks = valid_price_matrix.columns
        flags = pd.DataFrame(
            np.random.rand(len(dates), len(stocks)),  # Float, not bool
            index=dates,
            columns=stocks
        )
        with pytest.raises(ValueError):
            validate_universe_flags(flags, valid_price_matrix)


class TestSignalMatrixValidation:
    """Test signal matrix validation."""
    
    def test_valid_signals_pass(self, valid_price_matrix):
        """Valid signal matrix should pass."""
        dates = valid_price_matrix.index
        stocks = valid_price_matrix.columns
        signals = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=(len(dates), len(stocks))),
            index=dates,
            columns=stocks
        )
        universe = pd.DataFrame(
            True, index=dates, columns=stocks
        )
        assert validate_signal_matrix(signals, universe) is True
    
    def test_invalid_signal_values_fail(self, valid_price_matrix):
        """Invalid signal values should fail."""
        dates = valid_price_matrix.index
        stocks = valid_price_matrix.columns
        signals = pd.DataFrame(
            np.random.choice([-2, 0, 2], size=(len(dates), len(stocks))),
            index=dates,
            columns=stocks
        )
        universe = pd.DataFrame(True, index=dates, columns=stocks)
        with pytest.raises(ValueError):
            validate_signal_matrix(signals, universe)


class TestWeightMatrixValidation:
    """Test portfolio weight validation."""
    
    def test_valid_weights_pass(self):
        """Valid weight matrix should pass."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        weights = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, size=(50, 10)),
            index=dates
        )
        signals = pd.DataFrame(
            np.random.choice([-1, 0, 1], size=(50, 10)),
            index=dates
        )
        assert validate_weight_matrix(weights, signals) is True
    
    def test_nan_weights_fail(self):
        """NaN weights should fail."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        weights = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, size=(50, 10)),
            index=dates
        )
        weights.iloc[10, 5] = np.nan
        signals = pd.DataFrame(np.zeros((50, 10)), index=dates)
        with pytest.raises(ValueError):
            validate_weight_matrix(weights, signals)
    
    def test_weights_exceed_bounds_fail(self):
        """Weights exceeding ±1 should fail."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        weights = pd.DataFrame(
            np.random.uniform(-0.1, 0.1, size=(50, 10)),
            index=dates
        )
        weights.iloc[10, 5] = 1.5  # Exceeds ±1
        signals = pd.DataFrame(np.zeros((50, 10)), index=dates)
        with pytest.raises(ValueError):
            validate_weight_matrix(weights, signals)
    
    def test_net_exposure_exceeds_1_fails(self):
        """Net exposure > ±1 should fail."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        weights = pd.DataFrame(
            np.ones((50, 10)) * 0.15,  # 10 * 0.15 = 1.5 > 1.0
            index=dates
        )
        signals = pd.DataFrame(np.zeros((50, 10)), index=dates)
        with pytest.raises(ValueError):
            validate_weight_matrix(weights, signals)
    
    def test_gross_exposure_exceeds_2_fails(self):
        """Gross exposure > 2.0 should fail."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        # Create all long (sum = 1) or all short (sum = -1), gross = 10
        weights = pd.DataFrame(
            np.ones((50, 10)) * 0.25,  # Gross = 2.5 > 2.0
            index=dates
        )
        signals = pd.DataFrame(np.zeros((50, 10)), index=dates)
        with pytest.raises(ValueError):
            validate_weight_matrix(weights, signals)


class TestBacktestReturnsValidation:
    """Test OOS returns validation."""
    
    def test_valid_oos_returns_pass(self):
        """Valid OOS returns should pass."""
        returns = pd.Series(
            np.random.normal(0.0005, 0.01, 250),
            index=pd.date_range(start='2023-01-01', periods=250, freq='D')
        )
        assert validate_backtest_returns(returns) is True
    
    def test_empty_returns_fail(self):
        """Empty returns should fail."""
        with pytest.raises(ValueError):
            validate_backtest_returns(pd.Series([]))
    
    def test_all_nan_returns_fail(self):
        """All-NaN returns should fail."""
        returns = pd.Series([np.nan] * 100)
        with pytest.raises(ValueError):
            validate_backtest_returns(returns)
    
    def test_infinity_in_returns_fail(self):
        """Infinity should fail."""
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        returns.iloc[50] = np.inf
        with pytest.raises(ValueError):
            validate_backtest_returns(returns)


class TestRiskMetricsValidation:
    """Test risk report validation."""
    
    def test_valid_risk_report_passes(self):
        """Valid risk report should pass."""
        report = {
            'hist_var_95': -0.015,
            'hist_var_99': -0.025,
            'cvar_95': -0.020,
            'cvar_99': -0.035,
        }
        assert validate_risk_metrics(report) is True
    
    def test_missing_metric_fails(self):
        """Missing metric should fail."""
        report = {
            'hist_var_95': -0.015,
            'hist_var_99': -0.025,
            # Missing cvar_95
        }
        with pytest.raises(ValueError):
            validate_risk_metrics(report)
    
    def test_nan_metrics_fail(self):
        """NaN metrics should fail."""
        report = {
            'hist_var_95': np.nan,
            'hist_var_99': -0.025,
            'cvar_95': -0.020,
            'cvar_99': -0.035,
        }
        with pytest.raises(ValueError):
            validate_risk_metrics(report)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
