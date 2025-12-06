"""
Feature engineering pipeline for Prometheus AI.

This module provides data transformation and feature engineering
for price prediction and portfolio optimization models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from prometheus.features.technical_indicators import TechnicalIndicators


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    
    window_size: int = 30
    price_features: bool = True
    volume_features: bool = True
    onchain_features: bool = True
    macro_features: bool = True
    technical_features: bool = True
    
    # Scaling options
    scaling_method: str = "minmax"  # "minmax" or "standard"
    scale_range: Tuple[float, float] = (0.0, 1.0)
    
    # Feature selection
    feature_columns: List[str] = field(default_factory=list)
    exclude_columns: List[str] = field(default_factory=list)


class FeatureEngineer:
    """
    Feature engineering pipeline for Prometheus AI.
    
    Generates features from:
    1. Price data (OHLCV)
    2. Technical indicators (RSI, MACD, Bollinger)
    3. On-chain metrics (volume, whale movements)
    4. Macro factors (inflation, interest rates)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration.
        """
        self.config = config or FeatureConfig()
        self.technical_indicators = TechnicalIndicators()
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        self._is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit the feature engineering pipeline.
        
        Args:
            data: Training data DataFrame.
            
        Returns:
            Self for method chaining.
        """
        # Generate features for fitting
        features = self.engineer_features(data)
        
        # Fit scaler
        if self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler(feature_range=self.config.scale_range)
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(features.values)
        self.feature_names = features.columns.tolist()
        self._is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Scaled feature array.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        
        features = self.engineer_features(data)
        
        # Ensure same columns
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0
        
        features = features[self.feature_names]
        
        return self.scaler.transform(features.values)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Scaled feature array.
        """
        self.fit(data)
        return self.transform(data)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw data.
        
        Args:
            data: Raw input DataFrame with price data.
            
        Returns:
            DataFrame with engineered features.
        """
        features = pd.DataFrame(index=data.index)
        
        if self.config.price_features:
            price_feats = self._price_features(data)
            features = pd.concat([features, price_feats], axis=1)
        
        if self.config.volume_features:
            volume_feats = self._volume_features(data)
            features = pd.concat([features, volume_feats], axis=1)
        
        if self.config.technical_features:
            tech_feats = self._technical_features(data)
            features = pd.concat([features, tech_feats], axis=1)
        
        if self.config.onchain_features:
            onchain_feats = self._onchain_features(data)
            features = pd.concat([features, onchain_feats], axis=1)
        
        if self.config.macro_features:
            macro_feats = self._macro_features(data)
            features = pd.concat([features, macro_feats], axis=1)
        
        # Remove excluded columns
        for col in self.config.exclude_columns:
            if col in features.columns:
                features = features.drop(columns=[col])
        
        # Handle infinities and NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)
        
        return features
    
    def _price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        df = pd.DataFrame(index=data.index)
        
        close = data.get("close", data.iloc[:, 0])
        
        # Returns at various horizons
        df["return_1d"] = close.pct_change(1)
        df["return_7d"] = close.pct_change(7)
        df["return_30d"] = close.pct_change(30)
        
        # Log returns
        df["log_return_1d"] = np.log(close / close.shift(1))
        
        # Moving average ratios
        df["sma_7_ratio"] = close / close.rolling(7).mean()
        df["sma_30_ratio"] = close / close.rolling(30).mean()
        df["ema_7_ratio"] = close / close.ewm(span=7).mean()
        
        # Volatility
        df["volatility_7d"] = close.pct_change().rolling(7).std()
        df["volatility_30d"] = close.pct_change().rolling(30).std()
        
        # Price momentum
        df["momentum_7d"] = close - close.shift(7)
        df["momentum_30d"] = close - close.shift(30)
        
        # High/Low features if available
        if "high" in data.columns and "low" in data.columns:
            df["price_range"] = (data["high"] - data["low"]) / close
            df["close_to_high"] = (data["high"] - close) / (data["high"] - data["low"] + 1e-10)
        
        return df
    
    def _volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        df = pd.DataFrame(index=data.index)
        
        if "volume" not in data.columns:
            return df
        
        volume = data["volume"]
        close = data.get("close", data.iloc[:, 0])
        
        # Volume ratios
        df["volume_sma_7_ratio"] = volume / volume.rolling(7).mean()
        df["volume_sma_30_ratio"] = volume / volume.rolling(30).mean()
        df["volume_change"] = volume.pct_change()
        
        # Volume-price correlation
        df["vp_corr_7d"] = close.rolling(7).corr(volume)
        df["vp_corr_30d"] = close.rolling(30).corr(volume)
        
        # Volume momentum
        df["volume_momentum"] = volume - volume.shift(1)
        
        # Relative volume
        df["relative_volume"] = volume / volume.rolling(20).mean()
        
        return df
    
    def _technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features."""
        df = pd.DataFrame(index=data.index)
        
        close = data.get("close", data.iloc[:, 0])
        high = data.get("high", close)
        low = data.get("low", close)
        
        # RSI
        df["rsi_14"] = self.technical_indicators.calculate_rsi(close, 14)
        df["rsi_normalized"] = df["rsi_14"] / 100
        
        # RSI signals
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)
        
        # MACD
        macd, signal, hist = self.technical_indicators.calculate_macd(close)
        df["macd"] = macd / close  # Normalize by price
        df["macd_signal"] = signal / close
        df["macd_histogram"] = hist / close
        df["macd_crossover"] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(float)
        
        # Bollinger Bands
        upper, middle, lower = self.technical_indicators.calculate_bollinger_bands(close)
        df["bb_upper_ratio"] = upper / close
        df["bb_lower_ratio"] = lower / close
        df["bb_width"] = (upper - lower) / middle
        df["bb_position"] = (close - lower) / (upper - lower + 1e-10)
        
        # Stochastic
        stoch_k, stoch_d = self.technical_indicators.calculate_stochastic(high, low, close)
        df["stoch_k"] = stoch_k / 100
        df["stoch_d"] = stoch_d / 100
        
        return df
    
    def _onchain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate on-chain metric features."""
        df = pd.DataFrame(index=data.index)
        
        # Whale transactions
        if "whale_transactions" in data.columns:
            whale_tx = data["whale_transactions"]
            df["whale_tx_sma"] = whale_tx.rolling(7).mean()
            df["whale_tx_change"] = whale_tx.pct_change()
            df["whale_tx_z"] = (whale_tx - whale_tx.rolling(30).mean()) / (
                whale_tx.rolling(30).std() + 1e-10
            )
        
        # Active addresses
        if "active_addresses" in data.columns:
            active = data["active_addresses"]
            df["active_addr_sma"] = active.rolling(7).mean()
            df["active_addr_change"] = active.pct_change()
            df["active_addr_ratio"] = active / active.rolling(30).mean()
        
        # Exchange flows
        if "exchange_inflow" in data.columns:
            inflow = data["exchange_inflow"]
            outflow = data.get("exchange_outflow", pd.Series(0, index=data.index))
            df["exchange_netflow"] = inflow - outflow
            df["exchange_netflow_ratio"] = df["exchange_netflow"] / (inflow + 1e-10)
        
        # Hash rate (for PoW chains)
        if "hash_rate" in data.columns:
            df["hash_rate_change"] = data["hash_rate"].pct_change()
        
        return df
    
    def _macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate macro-economic features."""
        df = pd.DataFrame(index=data.index)
        
        # Inflation rate
        if "inflation_rate" in data.columns:
            df["inflation"] = data["inflation_rate"]
            df["inflation_change"] = data["inflation_rate"].diff()
        
        # Federal Reserve rate
        if "fed_rate" in data.columns:
            df["fed_rate"] = data["fed_rate"]
            df["fed_rate_change"] = data["fed_rate"].diff()
        
        # Dollar index
        if "dxy_index" in data.columns:
            df["dxy"] = data["dxy_index"].pct_change()
            df["dxy_sma"] = data["dxy_index"].rolling(7).mean() / data["dxy_index"]
        
        # Gold/commodities correlation
        if "gold_price" in data.columns and "close" in data.columns:
            df["gold_corr"] = data["close"].rolling(30).corr(data["gold_price"])
        
        # VIX (volatility index)
        if "vix" in data.columns:
            df["vix"] = data["vix"]
            df["vix_change"] = data["vix"].pct_change()
        
        return df
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction.
        
        Args:
            features: Feature array of shape (n_samples, n_features).
            targets: Target array of shape (n_samples,) or (n_samples, n_targets).
            sequence_length: Length of input sequences.
            
        Returns:
            Tuple of (X, y) where X has shape (n_sequences, sequence_length, n_features).
        """
        n_samples = len(features)
        n_sequences = n_samples - sequence_length
        
        if n_sequences <= 0:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length {sequence_length}")
        
        n_features = features.shape[1] if features.ndim > 1 else 1
        
        X = np.zeros((n_sequences, sequence_length, n_features))
        y = np.zeros((n_sequences,) + targets.shape[1:] if targets.ndim > 1 else (n_sequences,))
        
        for i in range(n_sequences):
            X[i] = features[i:i + sequence_length]
            y[i] = targets[i + sequence_length]
        
        return X, y
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        feature_index: int = 0,
    ) -> np.ndarray:
        """
        Inverse transform scaled predictions.
        
        Args:
            predictions: Scaled prediction values.
            feature_index: Index of the feature to inverse transform.
            
        Returns:
            Original-scale predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted first")
        
        if isinstance(self.scaler, MinMaxScaler):
            scale = self.scaler.scale_[feature_index]
            min_val = self.scaler.min_[feature_index]
            return (predictions - min_val) / scale
        else:  # StandardScaler
            mean = self.scaler.mean_[feature_index]
            std = self.scaler.scale_[feature_index]
            return predictions * std + mean
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance from a fitted model.
        
        Args:
            model: Fitted model with feature_importances_ attribute.
            feature_names: Optional custom feature names.
            
        Returns:
            DataFrame with feature importance scores.
        """
        names = feature_names or self.feature_names
        
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            raise ValueError("Model does not have feature_importances_ attribute")
        
        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
