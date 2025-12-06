"""
Tests for Prometheus AI models.

Run with: pytest tests/ -v
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ============================================================================
# LSTM Predictor Tests
# ============================================================================


class TestLSTMPredictor:
    """Tests for LSTM price prediction model."""
    
    @pytest.fixture
    def model(self):
        """Create LSTM predictor instance."""
        # Import inside fixture to handle missing dependencies gracefully
        try:
            from prometheus.models.lstm_predictor import LSTMPredictor
            return LSTMPredictor(
                input_dim=64,
                hidden_dim=128,
                num_layers=2,
                output_horizon=7,
                dropout=0.1,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_model_initialization(self, model) -> None:
        """Test model initializes with correct parameters."""
        assert model.input_dim == 64
        assert model.hidden_dim == 128
        assert model.output_horizon == 7
    
    def test_forward_shape(self, model) -> None:
        """Test forward pass produces correct output shapes."""
        import torch
        
        batch_size = 32
        seq_len = 30
        input_dim = 64
        
        x = torch.randn(batch_size, seq_len, input_dim)
        predictions, confidence = model(x)
        
        assert predictions.shape == (batch_size, 7)
        assert confidence.shape == (batch_size, 7)
    
    def test_confidence_range(self, model) -> None:
        """Test confidence scores are in valid range [0, 1]."""
        import torch
        
        x = torch.randn(16, 30, 64)
        _, confidence = model(x)
        
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
    
    def test_gradient_flow(self, model) -> None:
        """Test gradients flow properly through the model."""
        import torch
        
        x = torch.randn(8, 30, 64, requires_grad=True)
        predictions, _ = model(x)
        
        loss = predictions.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_predict_method(self, model) -> None:
        """Test predict method with numpy input."""
        historical_data = np.random.randn(30, 64).astype(np.float32)
        
        predictions, confidence = model.predict(historical_data, device="cpu")
        
        assert predictions.shape == (1, 7)
        assert confidence.shape == (1, 7)
    
    def test_batch_predict(self, model) -> None:
        """Test prediction with batched input."""
        batch_data = np.random.randn(5, 30, 64).astype(np.float32)
        
        predictions, confidence = model.predict(batch_data, device="cpu")
        
        assert predictions.shape == (5, 7)
        assert confidence.shape == (5, 7)


# ============================================================================
# Transformer Model Tests
# ============================================================================


class TestTransformerForecaster:
    """Tests for Transformer forecasting model."""
    
    @pytest.fixture
    def model(self):
        """Create Transformer forecaster instance."""
        try:
            from prometheus.models.transformer_model import TransformerForecaster
            return TransformerForecaster(
                input_dim=64,
                d_model=128,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=1,
                output_horizon=7,
            )
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_forward_shape(self, model) -> None:
        """Test forward pass output shapes."""
        import torch
        
        batch_size = 16
        seq_len = 30
        
        x = torch.randn(batch_size, seq_len, 64)
        predictions, confidence, volatility = model(x)
        
        assert predictions.shape == (batch_size, 7)
        assert confidence.shape == (batch_size, 7)
        assert volatility.shape == (batch_size, 7)
    
    def test_predict_method(self, model) -> None:
        """Test predict method returns dictionary."""
        data = np.random.randn(30, 64).astype(np.float32)
        
        result = model.predict(data, device="cpu")
        
        assert "predictions" in result
        assert "confidence" in result
        assert "volatility" in result


# ============================================================================
# Q-Learning Portfolio Optimizer Tests
# ============================================================================


class TestPortfolioOptimizer:
    """Tests for Q-Learning portfolio optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer instance."""
        try:
            from prometheus.models.q_learning import PortfolioOptimizer
            return PortfolioOptimizer(state_dim=32)
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    @pytest.fixture
    def portfolio_state(self):
        """Create sample portfolio state."""
        from prometheus.models.q_learning import PortfolioState
        return PortfolioState(
            total_value=100000,
            asset_allocation={"BTC": 0.5, "ETH": 0.3, "USDC": 0.2},
            unrealized_pnl=0,
            volatility_30d=0.2,
            sharpe_ratio=1.5,
            market_regime="bull",
        )
    
    @pytest.fixture
    def risk_profile(self):
        """Create sample risk profile."""
        from prometheus.models.q_learning import RiskProfile
        return RiskProfile(
            risk_tolerance=0.5,
            time_to_goal=24,
            investment_amount=100000,
            monthly_contribution=1000,
        )
    
    def test_action_selection(self, optimizer) -> None:
        """Test action selection returns valid action."""
        from prometheus.models.q_learning import Action
        
        state = np.random.randn(32).astype(np.float32)
        action = optimizer.select_action(state, training=False)
        
        assert isinstance(action, Action)
        assert action.value in range(7)
    
    def test_exploration(self, optimizer) -> None:
        """Test epsilon-greedy exploration."""
        optimizer.epsilon = 1.0  # Force exploration
        
        state = np.random.randn(32).astype(np.float32)
        actions = [optimizer.select_action(state, training=True) for _ in range(100)]
        
        # Should have variety in actions due to exploration
        unique_actions = len(set(a.value for a in actions))
        assert unique_actions > 1
    
    def test_reward_calculation(
        self,
        optimizer,
        portfolio_state,
        risk_profile,
    ) -> None:
        """Test reward calculation for portfolio changes."""
        from prometheus.models.q_learning import PortfolioState
        
        new_portfolio = PortfolioState(
            total_value=105000,  # 5% gain
            asset_allocation={"BTC": 0.4, "ETH": 0.3, "USDC": 0.3},
            unrealized_pnl=5000,
            volatility_30d=0.15,
            sharpe_ratio=1.8,
            market_regime="bull",
        )
        
        reward = optimizer.calculate_reward(
            portfolio_state,
            new_portfolio,
            risk_profile,
        )
        
        # Positive return should give positive reward
        assert reward > 0
    
    def test_negative_reward(
        self,
        optimizer,
        portfolio_state,
        risk_profile,
    ) -> None:
        """Test negative reward for losses."""
        from prometheus.models.q_learning import PortfolioState
        
        new_portfolio = PortfolioState(
            total_value=90000,  # 10% loss
            asset_allocation={"BTC": 0.5, "ETH": 0.3, "USDC": 0.2},
            unrealized_pnl=-10000,
            volatility_30d=0.35,  # Higher volatility
            sharpe_ratio=0.5,
            market_regime="bear",
        )
        
        reward = optimizer.calculate_reward(
            portfolio_state,
            new_portfolio,
            risk_profile,
        )
        
        # Loss should give negative or lower reward
        assert reward < 5  # Less than a significant positive
    
    def test_state_encoding(
        self,
        optimizer,
        portfolio_state,
        risk_profile,
    ) -> None:
        """Test state encoding produces correct shape."""
        market_data = {
            "btc_return_7d": 0.05,
            "eth_return_7d": 0.03,
            "market_volatility": 0.2,
            "fear_greed_index": 60,
        }
        
        state = optimizer.encode_state(portfolio_state, risk_profile, market_data)
        
        assert state.shape == (32,)
        assert state.dtype == np.float32
    
    def test_rebalance_recommendation(
        self,
        optimizer,
        portfolio_state,
        risk_profile,
    ) -> None:
        """Test rebalance recommendation generation."""
        market_data = {
            "btc_return_7d": 0.02,
            "eth_return_7d": 0.03,
            "market_volatility": 0.2,
            "fear_greed_index": 55,
        }
        
        result = optimizer.get_rebalance_recommendation(
            portfolio_state,
            risk_profile,
            market_data,
        )
        
        assert "action" in result
        assert "recommendations" in result
        assert "confidence" in result
        assert "risk_assessment" in result
    
    def test_replay_buffer(self, optimizer) -> None:
        """Test experience replay buffer."""
        for i in range(100):
            state = np.random.randn(32).astype(np.float32)
            action = i % 7
            reward = np.random.randn()
            next_state = np.random.randn(32).astype(np.float32)
            done = False
            
            optimizer.replay_buffer.push(state, action, reward, next_state, done)
        
        assert len(optimizer.replay_buffer) == 100
        
        # Sample batch
        states, actions, rewards, next_states, dones = optimizer.replay_buffer.sample(32)
        
        assert states.shape == (32, 32)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
    
    def test_train_step(self, optimizer) -> None:
        """Test training step executes without error."""
        # Fill buffer with experiences
        for i in range(100):
            state = np.random.randn(32).astype(np.float32)
            action = i % 7
            reward = np.random.randn()
            next_state = np.random.randn(32).astype(np.float32)
            done = False
            
            optimizer.replay_buffer.push(state, action, reward, next_state, done)
        
        loss = optimizer.train_step()
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)


# ============================================================================
# Technical Indicators Tests
# ============================================================================


class TestTechnicalIndicators:
    """Tests for technical indicators."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        import pandas as pd
        
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(n) * 2),
            "high": 100 + np.cumsum(np.random.randn(n) * 2) + 2,
            "low": 100 + np.cumsum(np.random.randn(n) * 2) - 2,
            "volume": np.random.randint(1000, 10000, n),
        })
    
    def test_rsi_calculation(self, sample_data) -> None:
        """Test RSI calculation."""
        try:
            from prometheus.features.technical_indicators import TechnicalIndicators
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        indicators = TechnicalIndicators()
        rsi = indicators.calculate_rsi(sample_data["close"], period=14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self, sample_data) -> None:
        """Test MACD calculation."""
        try:
            from prometheus.features.technical_indicators import TechnicalIndicators
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        indicators = TechnicalIndicators()
        macd, signal, histogram = indicators.calculate_macd(sample_data["close"])
        
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)
    
    def test_bollinger_bands(self, sample_data) -> None:
        """Test Bollinger Bands calculation."""
        try:
            from prometheus.features.technical_indicators import TechnicalIndicators
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        indicators = TechnicalIndicators()
        upper, middle, lower = indicators.calculate_bollinger_bands(
            sample_data["close"],
            period=20,
        )
        
        # Upper should be above lower
        valid_idx = upper.notna() & lower.notna()
        assert (upper[valid_idx] >= lower[valid_idx]).all()
    
    def test_calculate_all(self, sample_data) -> None:
        """Test calculating all indicators."""
        try:
            from prometheus.features.technical_indicators import TechnicalIndicators
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        indicators = TechnicalIndicators()
        result = indicators.calculate_all(sample_data)
        
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "stoch_k" in result.columns


# ============================================================================
# Feature Pipeline Tests
# ============================================================================


class TestFeaturePipeline:
    """Tests for feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        import pandas as pd
        
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            "open": 100 + np.cumsum(np.random.randn(n) * 2),
            "high": 100 + np.cumsum(np.random.randn(n) * 2) + 2,
            "low": 100 + np.cumsum(np.random.randn(n) * 2) - 2,
            "close": 100 + np.cumsum(np.random.randn(n) * 2),
            "volume": np.random.randint(1000, 10000, n),
        })
    
    def test_feature_engineering(self, sample_data) -> None:
        """Test feature engineering produces expected features."""
        try:
            from prometheus.features.pipeline import FeatureEngineer, FeatureConfig
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        config = FeatureConfig(
            price_features=True,
            volume_features=True,
            technical_features=True,
        )
        
        engineer = FeatureEngineer(config)
        features = engineer.engineer_features(sample_data)
        
        assert "return_1d" in features.columns
        assert "rsi_14" in features.columns
        assert "macd" in features.columns
    
    def test_fit_transform(self, sample_data) -> None:
        """Test fit_transform method."""
        try:
            from prometheus.features.pipeline import FeatureEngineer
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        engineer = FeatureEngineer()
        scaled = engineer.fit_transform(sample_data)
        
        assert isinstance(scaled, np.ndarray)
        assert len(scaled) > 0
    
    def test_sequence_creation(self, sample_data) -> None:
        """Test sequence creation for time series."""
        try:
            from prometheus.features.pipeline import FeatureEngineer
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_data)
        
        X, y = engineer.create_sequences(
            features.values,
            sample_data["close"].values,
            sequence_length=30,
        )
        
        assert X.shape[1] == 30
        assert len(X) == len(y)


# ============================================================================
# Risk Profile Validation Tests
# ============================================================================


class TestRiskProfile:
    """Tests for risk profile validation."""
    
    def test_valid_risk_profile(self) -> None:
        """Test valid risk profile creation."""
        try:
            from prometheus.models.q_learning import RiskProfile
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        profile = RiskProfile(
            risk_tolerance=0.5,
            time_to_goal=24,
            investment_amount=100000,
            monthly_contribution=1000,
        )
        
        assert profile.risk_tolerance == 0.5
        assert profile.time_to_goal == 24
    
    def test_invalid_risk_tolerance(self) -> None:
        """Test that invalid risk tolerance raises error."""
        try:
            from prometheus.models.q_learning import RiskProfile
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        with pytest.raises(ValueError):
            RiskProfile(
                risk_tolerance=1.5,  # Invalid: > 1
                time_to_goal=24,
                investment_amount=100000,
                monthly_contribution=1000,
            )
    
    def test_invalid_time_to_goal(self) -> None:
        """Test that invalid time_to_goal raises error."""
        try:
            from prometheus.models.q_learning import RiskProfile
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        with pytest.raises(ValueError):
            RiskProfile(
                risk_tolerance=0.5,
                time_to_goal=0,  # Invalid: < 1
                investment_amount=100000,
                monthly_contribution=1000,
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_prediction_pipeline(self) -> None:
        """Test complete prediction pipeline."""
        try:
            import pandas as pd
            from prometheus.models.lstm_predictor import LSTMPredictor
            from prometheus.features.pipeline import FeatureEngineer
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        # Create sample data
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(n) * 2),
            "volume": np.random.randint(1000, 10000, n),
        })
        
        # Feature engineering
        engineer = FeatureEngineer()
        features = engineer.fit_transform(data)
        
        # Create sequences
        X, y = engineer.create_sequences(features, data["close"].values, 30)
        
        # Model prediction (use last sequence)
        model = LSTMPredictor(input_dim=X.shape[2], hidden_dim=64, output_horizon=7)
        predictions, confidence = model.predict(X[-1:], device="cpu")
        
        assert predictions.shape == (1, 7)
        assert confidence.shape == (1, 7)
    
    def test_optimization_pipeline(self) -> None:
        """Test complete portfolio optimization pipeline."""
        try:
            from prometheus.models.q_learning import (
                PortfolioOptimizer,
                PortfolioState,
                RiskProfile,
            )
        except ImportError:
            pytest.skip("Dependencies not installed")
        
        # Create optimizer
        optimizer = PortfolioOptimizer(state_dim=32)
        
        # Create portfolio and profile
        portfolio = PortfolioState(
            total_value=100000,
            asset_allocation={"BTC": 0.5, "ETH": 0.3, "USDC": 0.2},
            unrealized_pnl=0,
            volatility_30d=0.2,
            sharpe_ratio=1.5,
            market_regime="sideways",
        )
        
        profile = RiskProfile(
            risk_tolerance=0.6,
            time_to_goal=36,
            investment_amount=100000,
            monthly_contribution=500,
        )
        
        market_data = {
            "btc_return_7d": 0.01,
            "eth_return_7d": 0.02,
            "market_volatility": 0.15,
            "fear_greed_index": 45,
        }
        
        # Get recommendation
        result = optimizer.get_rebalance_recommendation(
            portfolio,
            profile,
            market_data,
        )
        
        assert "action" in result
        assert "recommendations" in result
        assert "risk_assessment" in result
        assert result["risk_assessment"]["risk_score"] >= 0
