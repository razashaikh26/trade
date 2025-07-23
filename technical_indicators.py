import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

class TechnicalIndicators:
    """
    Advanced technical indicators for dynamic trading strategies
    """
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def atr_based_levels(current_price: float, atr_value: float, multiplier: float = 2.0, testnet: bool = False) -> Dict[str, float]:
        """
        Calculate dynamic stop loss and take profit levels based on ATR
        
        Args:
            current_price: Current market price
            atr_value: Current ATR value
            multiplier: ATR multiplier for SL/TP distance
            testnet: If True, use half of the main ATR distance for testing
        
        Returns:
            Dictionary with dynamic levels
        """
        atr_distance = atr_value * multiplier
        
        if testnet:
            # Use half of the main ATR distance for testnet
            atr_distance = atr_distance * 0.5
        
        return {
            'long_stop_loss': current_price - atr_distance,
            'long_take_profit': current_price + (atr_distance * 1.5),  # 1.5:1 risk/reward
            'short_stop_loss': current_price + atr_distance,
            'short_take_profit': current_price - (atr_distance * 1.5)
        }
    
    @staticmethod
    def atr_based_levels_for_position(entry_price: float, atr_value: float, multiplier: float = 2.0, testnet: bool = False) -> Dict[str, float]:
        """
        Calculate dynamic stop loss and take profit levels based on entry price and ATR
        
        Args:
            entry_price: Position entry price
            atr_value: Current ATR value
            multiplier: ATR multiplier for SL/TP distance
            testnet: If True, use half of the main ATR distance for testing
        
        Returns:
            Dictionary with dynamic levels based on entry price
        """
        atr_distance = atr_value * multiplier
        
        if testnet:
            # Use half of the main ATR distance for testnet
            atr_distance = atr_distance * 0.5
        
        return {
            'long_stop_loss': entry_price - atr_distance,
            'long_take_profit': entry_price + (atr_distance * 1.5),  # 1.5:1 risk/reward
            'short_stop_loss': entry_price + atr_distance,
            'short_take_profit': entry_price - (atr_distance * 1.5)
        }
    
    @staticmethod
    def detect_engulfing_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect bullish and bearish engulfing candlestick patterns
        
        Returns:
            Dictionary indicating presence of engulfing patterns
        """
        if len(df) < 2:
            return {'bullish_engulfing': False, 'bearish_engulfing': False}
        
        # Get last two candles
        prev_candle = df.iloc[-2]
        current_candle = df.iloc[-1]
        
        # Previous candle data
        prev_open = prev_candle['open']
        prev_close = prev_candle['close']
        prev_high = prev_candle['high']
        prev_low = prev_candle['low']
        
        # Current candle data
        curr_open = current_candle['open']
        curr_close = current_candle['close']
        curr_high = current_candle['high']
        curr_low = current_candle['low']
        
        # Bullish Engulfing Pattern
        # 1. Previous candle is bearish (red)
        # 2. Current candle is bullish (green)
        # 3. Current candle's body completely engulfs previous candle's body
        bullish_engulfing = (
            prev_close < prev_open and  # Previous candle is bearish
            curr_close > curr_open and  # Current candle is bullish
            curr_open < prev_close and  # Current open is below previous close
            curr_close > prev_open      # Current close is above previous open
        )
        
        # Bearish Engulfing Pattern
        # 1. Previous candle is bullish (green)
        # 2. Current candle is bearish (red)
        # 3. Current candle's body completely engulfs previous candle's body
        bearish_engulfing = (
            prev_close > prev_open and  # Previous candle is bullish
            curr_close < curr_open and  # Current candle is bearish
            curr_open > prev_close and  # Current open is above previous close
            curr_close < prev_open      # Current close is below previous open
        )
        
        return {
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing
        }
    
    @staticmethod
    def add_volume_confirmation(df, period=20):
        """
        Add volume confirmation to the dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            period (int): Period for volume moving average
            
        Returns:
            pd.DataFrame: DataFrame with volume confirmation signals
        """
        df = df.copy()
        
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        
        # Volume is confirmed if current volume > volume MA
        df['volume_confirmed'] = df['volume'] > df['volume_ma']
        
        return df
    
    @staticmethod
    def detect_candlestick_patterns(df):
        """
        Detect common candlestick patterns
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with pattern detection columns
        """
        df = df.copy()
        
        # Bullish patterns
        df['bullish_engulfing'] = (df['close'] > df['open']) & \
                                 (df['close'].shift(1) < df['open'].shift(1)) & \
                                 (df['close'] > df['open'].shift(1)) & \
                                 (df['open'] < df['close'].shift(1))
        
        df['hammer'] = (df['close'] > df['open']) & \
                      ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & \
                      ((df['high'] - df['close']) < (df['open'] - df['close']) * 0.3)
        
        # Bearish patterns
        df['bearish_engulfing'] = (df['close'] < df['open']) & \
                                 (df['close'].shift(1) > df['open'].shift(1)) & \
                                 (df['close'] < df['open'].shift(1)) & \
                                 (df['open'] > df['close'].shift(1))
        
        df['shooting_star'] = (df['open'] > df['close']) & \
                             ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) & \
                             ((df['close'] - df['low']) < (df['open'] - df['close']) * 0.3)
        
        return df
    
    @staticmethod
    def get_market_regime(df, short_period=20, long_period=50):
        """
        Determine market regime (trending/range-bound)
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            short_period (int): Short-term moving average period
            long_period (int): Long-term moving average period
            
        Returns:
            str: Market regime ('trending' or 'range')
        """
        df = df.copy()
        
        # Calculate moving averages
        df['sma_short'] = df['close'].rolling(window=short_period).mean()
        df['sma_long'] = df['close'].rolling(window=long_period).mean()
        
        # Calculate ATR for volatility
        df['atr'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
        
        # Simple trend detection
        price_std = df['close'].rolling(window=20).std()
        atr_ratio = df['atr'] / df['close'].rolling(window=20).mean()
        
        if atr_ratio.iloc[-1] > 0.01:  # Threshold for trending market
            return 'trending'
        return 'range'
    
    @staticmethod
    def calculate_volatility_adjusted_position_size(
        account_balance: float,
        base_risk_percent: float,
        current_atr: float,
        price: float,
        atr_period_avg: float,
        min_size: float = 20.0,
        max_size_multiplier: float = 2.0
    ) -> float:
        """
        Calculate position size adjusted for current market volatility
        
        Args:
            account_balance: Current account balance
            base_risk_percent: Base risk percentage from config
            current_atr: Current ATR value
            price: Current price
            atr_period_avg: Average ATR over longer period
            min_size: Minimum position size
            max_size_multiplier: Maximum multiplier for position size
        
        Returns:
            Adjusted position size
        """
        # Calculate base position size
        base_position_value = account_balance * (base_risk_percent / 100)
        
        # Calculate volatility ratio (current ATR vs average ATR)
        if atr_period_avg > 0:
            volatility_ratio = current_atr / atr_period_avg
        else:
            volatility_ratio = 1.0
        
        # Adjust position size inversely to volatility
        # Higher volatility = smaller position size
        # Lower volatility = larger position size (up to max_size_multiplier)
        if volatility_ratio > 1.0:
            # High volatility - reduce position size
            adjustment_factor = 1.0 / volatility_ratio
        else:
            # Low volatility - increase position size (capped)
            adjustment_factor = min(1.0 / volatility_ratio, max_size_multiplier)
        
        adjusted_position_value = base_position_value * adjustment_factor
        
        # Convert to position size in base currency
        position_size = adjusted_position_value / price
        
        # Ensure minimum position size
        position_size = max(position_size, min_size)
        
        return position_size
    
    @staticmethod
    def get_market_volatility_state(current_atr: float, atr_ma: float) -> str:
        """
        Determine current market volatility state
        
        Returns:
            'LOW', 'NORMAL', or 'HIGH' volatility
        """
        if atr_ma == 0:
            return 'NORMAL'
        
        ratio = current_atr / atr_ma
        
        if ratio < 0.8:
            return 'LOW'
        elif ratio > 1.2:
            return 'HIGH'
        else:
            return 'NORMAL'
    
    @staticmethod
    def calculate_dynamic_atr_multiplier(volatility_state: str) -> float:
        """
        Calculate dynamic ATR multiplier based on market volatility
        
        Args:
            volatility_state: Current volatility state ('LOW', 'NORMAL', 'HIGH')
        
        Returns:
            ATR multiplier for stop loss/take profit calculation
        """
        multipliers = {
            'LOW': 1.5,      # Tighter stops in low volatility
            'NORMAL': 2.0,   # Standard stops in normal volatility
            'HIGH': 2.5      # Wider stops in high volatility
        }
        
        return multipliers.get(volatility_state, 2.0)
