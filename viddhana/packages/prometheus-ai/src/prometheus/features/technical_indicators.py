"""
Technical indicators for feature engineering.

This module provides implementations of common technical indicators
used in financial analysis: RSI, MACD, Bollinger Bands, and more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    
    # RSI settings
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands settings
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Moving average settings
    sma_short: int = 7
    sma_medium: int = 30
    sma_long: int = 200
    ema_span: int = 7


class TechnicalIndicators:
    """
    Technical indicators calculator for financial time-series.
    
    Provides RSI, MACD, Bollinger Bands, moving averages, and
    volatility metrics commonly used in price prediction.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None) -> None:
        """
        Initialize technical indicators calculator.
        
        Args:
            config: Configuration for indicator parameters.
        """
        self.config = config or IndicatorConfig()
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Args:
            prices: Price series.
            period: Look-back period (default from config).
            
        Returns:
            RSI values as pandas Series.
        """
        period = period or self.config.rsi_period
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        
        Args:
            prices: Price series.
            fast_period: Fast EMA period.
            slow_period: Slow EMA period.
            signal_period: Signal line period.
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram).
        """
        fast = fast_period or self.config.macd_fast
        slow = slow_period or self.config.macd_slow
        signal = signal_period or self.config.macd_signal
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
        num_std: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle Band + (num_std * std)
        Lower Band = Middle Band - (num_std * std)
        
        Args:
            prices: Price series.
            period: Look-back period.
            num_std: Number of standard deviations.
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band).
        """
        period = period or self.config.bb_period
        num_std = num_std or self.config.bb_std
        
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_bollinger_width(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
        num_std: Optional[float] = None,
    ) -> pd.Series:
        """
        Calculate Bollinger Band width.
        
        Width = (Upper - Lower) / Middle
        
        Args:
            prices: Price series.
            period: Look-back period.
            num_std: Number of standard deviations.
            
        Returns:
            Bollinger Band width as pandas Series.
        """
        upper, middle, lower = self.calculate_bollinger_bands(
            prices, period, num_std
        )
        return (upper - lower) / (middle + 1e-10)
    
    def calculate_sma(
        self,
        prices: pd.Series,
        period: int,
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series.
            period: Look-back period.
            
        Returns:
            SMA values as pandas Series.
        """
        return prices.rolling(window=period).mean()
    
    def calculate_ema(
        self,
        prices: pd.Series,
        span: int,
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series.
            span: EMA span.
            
        Returns:
            EMA values as pandas Series.
        """
        return prices.ewm(span=span, adjust=False).mean()
    
    def calculate_volatility(
        self,
        prices: pd.Series,
        period: int = 30,
    ) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            prices: Price series.
            period: Look-back period.
            
        Returns:
            Volatility values as pandas Series.
        """
        returns = prices.pct_change()
        return returns.rolling(window=period).std()
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA(True Range, period)
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: Look-back period.
            
        Returns:
            ATR values as pandas Series.
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return true_range.ewm(span=period, adjust=False).mean()
    
    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA(%K, d_period)
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            k_period: %K period.
            d_period: %D period.
            
        Returns:
            Tuple of (%K, %D).
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Williams %R.
        
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: Look-back period.
            
        Returns:
            Williams %R values as pandas Series.
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    def calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close: Close prices.
            volume: Volume data.
            
        Returns:
            OBV values as pandas Series.
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        
        return (direction * volume).cumsum()
    
    def calculate_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            volume: Volume data.
            
        Returns:
            VWAP values as pandas Series.
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def calculate_all(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
            
        Returns:
            DataFrame with all calculated indicators.
        """
        result = pd.DataFrame(index=data.index)
        
        close = data["close"]
        high = data.get("high", close)
        low = data.get("low", close)
        volume = data.get("volume", pd.Series(1, index=data.index))
        
        # RSI
        result["rsi"] = self.calculate_rsi(close)
        
        # MACD
        macd, signal, hist = self.calculate_macd(close)
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_histogram"] = hist
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(close)
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower
        result["bb_width"] = self.calculate_bollinger_width(close)
        
        # Moving Averages
        result["sma_7"] = self.calculate_sma(close, 7)
        result["sma_30"] = self.calculate_sma(close, 30)
        result["ema_7"] = self.calculate_ema(close, 7)
        result["ema_21"] = self.calculate_ema(close, 21)
        
        # Volatility
        result["volatility_7d"] = self.calculate_volatility(close, 7)
        result["volatility_30d"] = self.calculate_volatility(close, 30)
        
        # ATR
        result["atr"] = self.calculate_atr(high, low, close)
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        
        # Williams %R
        result["williams_r"] = self.calculate_williams_r(high, low, close)
        
        # OBV
        result["obv"] = self.calculate_obv(close, volume)
        
        # VWAP
        result["vwap"] = self.calculate_vwap(high, low, close, volume)
        
        return result


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Convenience function for RSI calculation."""
    return TechnicalIndicators().calculate_rsi(prices, period)


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for MACD calculation."""
    return TechnicalIndicators().calculate_macd(prices, fast, slow, signal)


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Convenience function for Bollinger Bands calculation."""
    return TechnicalIndicators().calculate_bollinger_bands(prices, period, num_std)
