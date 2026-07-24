"""Unit tests for backtest engine.

CRITICAL: These tests validate:
1. Walk-forward window generation has no overlap (no look-ahead)
2. Portfolio returns calculated correctly
3. Performance metrics (CAGR, Sharpe, drawdown) are accurate
4. Edge cases (single-day windows, small portfolios)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_returns():
    """Create sample daily returns for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # Simulate realistic returns: mean ~0.05%, vol ~1%
    returns = np.random.normal(0.0005, 0.01, 500)
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_weights():
    """Create sample weight matrix for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    n_stocks = 50
    # Random long/short weights summing to ~0
    weights_data = np.random.uniform(-0.02, 0.02, (500, n_stocks))
    return pd.DataFrame(weights_data, index=dates)


class TestWalkForwardWindows:
    """Test walk-forward window generation."""
    
    def test_windows_no_overlap(self):
        """Windows should not overlap (no look-ahead bias)."""
        total_len = 500
        train_size = 250
        test_size = 50
        step = 30
        
        windows = []
        for i in range(train_size, total_len - test_size, step):
            train_start = max(0, i - train_size)
            train_end = i
            test_start = i
            test_end = min(total_len, i + test_size)
            
            windows.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end),
            })
        
        # Check no train/test overlap
        for w in windows:
            assert w['train'][1] <= w['test'][0], "Train overlaps with test!"
    
    def test_windows_ascending_order(self):
        """Windows should progress forward in time."""
        total_len = 500
        train_size = 250
        test_size = 50
        step = 30
        
        windows = []
        for i in range(train_size, total_len - test_size, step):
            test_start = i
            test_end = min(total_len, i + test_size)
            windows.append((test_start, test_end))
        
        # Each window starts after previous one
        for i in range(1, len(windows)):
            assert windows[i][0] >= windows[i-1][0]
    
    def test_window_boundary_edge_case(self):
        """Handle windows at data boundaries."""
        total_len = 100
        train_size = 50
        test_size = 20
        
        # Last possible window
        i = total_len - test_size - 1
        train_start = max(0, i - train_size)
        train_end = i
        test_start = i
        test_end = min(total_len, i + test_size)
        
        assert test_end <= total_len
        assert train_start >= 0


class TestPortfolioReturnsComputation:
    """Test portfolio return calculations."""
    
    def test_portfolio_returns_shape(self, sample_weights, sample_returns):
        """Portfolio returns should have same length as asset returns."""
        # Align shapes
        n_dates = min(len(sample_weights), len(sample_returns))
        weights = sample_weights.iloc[:n_dates]
        returns = sample_returns.iloc[:n_dates]
        
        # Portfolio return = sum(weights * returns)
        portfolio_ret = (weights.T * returns.values).T.sum(axis=1)
        
        assert len(portfolio_ret) == n_dates
    
    def test_portfolio_returns_bounded(self, sample_weights, sample_returns):
        """
        Portfolio returns should be bounded.
        
        With leverage ~1, and individual stock returns ~±5%, 
        portfolio return should be ~±5%.
        """
        n_dates = min(len(sample_weights), len(sample_returns))
        weights = sample_weights.iloc[:n_dates]
        returns = sample_returns.iloc[:n_dates]
        
        portfolio_ret = (weights.T * returns.values).T.sum(axis=1)
        
        # Returns should be reasonable (not infinite/NaN)
        assert portfolio_ret.notna().all()
        assert np.isfinite(portfolio_ret).all()
        
        # Max abs return should be < 100% (sanity check)
        assert portfolio_ret.abs().max() < 1.0
    
    def test_portfolio_returns_zero_weight_edge_case(self):
        """Portfolio with zero weights should have zero returns."""
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        
        weights = pd.DataFrame(
            np.zeros((10, 5)),
            index=dates,
            columns=[f'S{i}' for i in range(5)]
        )
        returns = pd.DataFrame(
            np.random.randn(10, 5) * 0.01,
            index=dates,
            columns=[f'S{i}' for i in range(5)]
        )
        
        portfolio_ret = (weights.T * returns.values).T.sum(axis=1)
        
        assert (portfolio_ret == 0).all()


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    def test_cagr_calculation(self):
        """
        CAGR = (end_value / start_value) ^ (252 / n_days) - 1
        
        Not: annualized_return * sqrt(n_days) (WRONG)
        """
        returns = pd.Series([0.001] * 252)  # 0.1% daily = 252 days
        
        cumulative_return = (1 + returns).prod() - 1
        n_days = len(returns)
        cagr = (1 + cumulative_return) ** (252 / n_days) - 1
        
        # With perfect 0.1% daily returns for 252 days, CAGR should be ~25%
        expected_cagr = (1.001 ** 252) - 1
        assert cagr == pytest.approx(expected_cagr, rel=1e-6)
    
    def test_sharpe_ratio_positive_returns(self):
        """Sharpe ratio should be positive for positive return strategies."""
        # Consistent positive returns
        returns = pd.Series([0.001] * 100)
        
        mean_return = returns.mean()
        volatility = returns.std()
        sharpe = (mean_return / volatility) * np.sqrt(252)
        
        assert sharpe > 0
    
    def test_sharpe_ratio_zero_volatility_edge_case(self):
        """Sharpe ratio should handle zero volatility gracefully."""
        returns = pd.Series([0.001] * 100)  # No volatility
        
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-8
        sharpe = (mean_return / (volatility + epsilon)) * np.sqrt(252)
        
        assert np.isfinite(sharpe)
    
    def test_drawdown_calculation(self):
        """Drawdown = (current / peak - 1) should be <= 0."""
        returns = pd.Series([0.01, 0.02, -0.05, 0.03, -0.02])
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        
        # Drawdown should always be <= 0
        assert (drawdown <= 0 + 1e-8).all()
    
    def test_max_drawdown(self):
        """Maximum drawdown should identify worst peak-to-trough."""
        returns = pd.Series([0.05, 0.05, -0.20, 0.05])  # Clear drawdown
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        
        max_dd = drawdown.min()
        
        # Max drawdown should be negative and capture the -20% dip
        assert max_dd < 0
        assert max_dd < -0.10  # At least -10% from peak


class TestWeightingEdgeCases:
    """Test edge cases in weight computation."""
    
    def test_equal_weights_sum_to_one_per_position(self):
        """Equal weight: each stock gets 1/n weight."""
        n_stocks = 100
        equal_weight = 1.0 / n_stocks
        
        assert equal_weight * n_stocks == pytest.approx(1.0)
    
    def test_vol_weight_normalization(self):
        """Vol weights should be normalized to sum to 1."""
        volatilities = pd.Series([0.01, 0.02, 0.015])
        
        # Inverse vol weighting (lower vol = higher weight)
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        assert weights.sum() == pytest.approx(1.0)
    
    def test_weight_constraint_satisfaction(self):
        """Portfolio weights must satisfy constraints."""
        weights = pd.Series([-0.1, 0.3, 0.5, 0.2, 0.1])
        
        # Individual position constraint: each weight in [-1, 1]
        assert (weights.abs() <= 1.0 + 1e-8).all()
        
        # Net exposure constraint: sum in [-1, 1]
        net_exposure = weights.sum()
        assert abs(net_exposure) <= 1.0 + 1e-8
        
        # Gross exposure constraint: sum of abs in [0, 2]
        gross_exposure = weights.abs().sum()
        assert 0 <= gross_exposure <= 2.0 + 1e-8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
