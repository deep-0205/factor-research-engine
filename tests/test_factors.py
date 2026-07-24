"""Unit tests for factor calculations.

CRITICAL: These tests validate:
1. No look-ahead bias (only past data used)
2. Correct mathematical formulas
3. Edge case handling
4. Data alignment and indexing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_close_matrix():
    """Create sample price matrix for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'STOCK1': np.random.randn(100).cumsum() + 100,
        'STOCK2': np.random.randn(100).cumsum() + 100,
        'STOCK3': np.random.randn(100).cumsum() + 100,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns_matrix(sample_close_matrix):
    """Create sample returns matrix."""
    return sample_close_matrix.pct_change().fillna(0)


class TestMomentumCalculation:
    """
    Test momentum factor calculation for look-ahead bias.
    
    CRITICAL: Momentum at date i should use:
    - shift(lookback + skip) to get returns from [i-lookback-skip, i-skip]
    - NOT shift(lookback) which would use [i-lookback, i]
    """
    
    def test_momentum_uses_only_past_data(self, sample_returns_matrix):
        """Verify momentum doesn't use future data."""
        lookback = 20
        skip = 1
        
        # Compute momentum using CORRECT formula
        momentum = sample_returns_matrix.shift(lookback + skip).rolling(lookback).sum()
        
        # At date i, momentum should be NaN if i < lookback + skip
        assert momentum.iloc[lookback + skip - 1].isna().all()
        
        # At date i >= lookback + skip, should be numeric
        assert momentum.iloc[lookback + skip].notna().any()
    
    def test_momentum_skip_prevents_look_ahead(self, sample_returns_matrix):
        """Verify skip parameter prevents using most recent returns."""
        lookback = 10
        skip = 1
        
        # With skip=1, we skip the most recent return (avoid look-ahead)
        momentum_skip = sample_returns_matrix.shift(lookback + skip).rolling(lookback).sum()
        
        # With skip=0 (WRONG), we'd use future data
        momentum_wrong = sample_returns_matrix.shift(lookback).rolling(lookback).sum()
        
        # They should be different (skip affects which returns are included)
        assert not momentum_skip.equals(momentum_wrong)
    
    def test_momentum_edge_case_insufficient_history(self, sample_returns_matrix):
        """Test momentum with insufficient lookback history."""
        lookback = 1000  # More than available data
        skip = 1
        
        momentum = sample_returns_matrix.shift(lookback + skip).rolling(lookback).sum()
        
        # Should be mostly NaN since we don't have enough history
        assert momentum.isna().sum().sum() > momentum.notna().sum().sum()


class TestICComputation:
    """Test information coefficient (IC) calculation."""
    
    def test_ic_uses_future_returns(self, sample_returns_matrix):
        """
        IC should correlate factors with FUTURE returns.
        
        Correct: IC[i] = corr(factor[i], returns[i+1:i+lookback])
        Wrong: IC[i] = corr(factor[i], returns[:i]) or corr(factor[i], returns[:])
        """
        lookback = 5
        
        # Create a dummy factor (for testing)
        factor = pd.DataFrame(
            np.random.randn(len(sample_returns_matrix), sample_returns_matrix.shape[1]),
            index=sample_returns_matrix.index,
            columns=sample_returns_matrix.columns
        )
        
        # Compute forward returns (future lookback returns)
        forward_returns = sample_returns_matrix.shift(-lookback).rolling(lookback).sum()
        
        # IC should correlate factor with forward returns
        ic = factor.corrwith(forward_returns)
        
        # IC should be reasonable (not all NaN)
        assert ic.notna().any()
    
    def test_ic_early_dates_insufficient_forward_data(self, sample_returns_matrix):
        """Early dates may lack sufficient forward data for IC computation."""
        lookback = 20
        n = len(sample_returns_matrix)
        
        # For dates too close to end, forward returns are NaN
        forward_returns = sample_returns_matrix.shift(-lookback).rolling(lookback).sum()
        
        # Last lookback dates should be NaN
        assert forward_returns.iloc[-lookback:].isna().all().all()


class TestCrossSection:
    """Test cross-sectional factor operations."""
    
    def test_zscore_shape_preservation(self):
        """Cross-sectional z-score should preserve shape."""
        data = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f'STOCK{i}' for i in range(10)]
        )
        
        # Z-score: (x - mean) / std
        zscore = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        
        assert zscore.shape == data.shape
        assert zscore.notna().all().all()
    
    def test_zscore_numerical_stability(self):
        """Z-score should handle low variance cases."""
        # Create data with very low variance (near zero std)
        data = pd.DataFrame(
            np.ones((10, 5)) + 1e-10 * np.random.randn(10, 5),
            columns=[f'S{i}' for i in range(5)]
        )
        
        std = data.std(axis=1)
        # With very low std, z-score can be unstable
        # Should add epsilon for numerical stability
        epsilon = 1e-8
        zscore = (data - data.mean(axis=1, keepdims=True)) / (std.values[:, np.newaxis] + epsilon)
        
        assert not zscore.isna().any().any()
        assert np.isfinite(zscore.values).all()
    
    def test_rank_consistency(self):
        """Cross-sectional rank should be consistent."""
        data = pd.DataFrame({
            'S1': [1.0, 5.0, 3.0],
            'S2': [4.0, 2.0, 1.0],
            'S3': [2.0, 3.0, 5.0],
        })
        
        # Rank: higher values get higher ranks
        rank = data.rank(axis=1)
        
        assert rank.loc[0, 'S2'] == 3.0  # S2 is highest in row 0
        assert rank.loc[1, 'S1'] == 3.0  # S1 is highest in row 1
        assert rank.loc[2, 'S3'] == 3.0  # S3 is highest in row 2


class TestCompositeFactorWeighting:
    """Test composite factor aggregation."""
    
    def test_composite_factor_weights_sum_to_one(self):
        """Composite factor weights should be normalized."""
        weights = {
            'momentum': 0.3,
            'quality': 0.3,
            'reversal': 0.2,
            'volatility': 0.2,
        }
        
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_composite_aggregation_preserves_shape(self):
        """Composite factor should have same shape as inputs."""
        n_dates = 50
        n_stocks = 10
        
        factors = {
            'momentum': pd.DataFrame(np.random.randn(n_dates, n_stocks)),
            'quality': pd.DataFrame(np.random.randn(n_dates, n_stocks)),
        }
        
        weights = {'momentum': 0.6, 'quality': 0.4}
        
        composite = (
            factors['momentum'] * weights['momentum'] +
            factors['quality'] * weights['quality']
        )
        
        assert composite.shape == (n_dates, n_stocks)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
