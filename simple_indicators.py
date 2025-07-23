"""
Simple Technical Indicators using Pandas
Replaces TA-Lib to avoid C library compilation issues
"""

import pandas as pd
import numpy as np

def RSI(close_prices, timeperiod=14):
    """Calculate RSI using simple pandas operations"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def SMA(close_prices, timeperiod=20):
    """Simple Moving Average"""
    return close_prices.rolling(window=timeperiod).mean().values

def EMA(close_prices, timeperiod=20):
    """Exponential Moving Average"""
    return close_prices.ewm(span=timeperiod).mean().values

def MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD calculation"""
    ema_fast = pd.Series(close_prices).ewm(span=fastperiod).mean()
    ema_slow = pd.Series(close_prices).ewm(span=slowperiod).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    sma = pd.Series(close_prices).rolling(window=timeperiod).mean()
    std = pd.Series(close_prices).rolling(window=timeperiod).std()
    upper_band = sma + (std * nbdevup)
    lower_band = sma - (std * nbdevdn)
    return upper_band.values, sma.values, lower_band.values

def STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)
    
    lowest_low = low_series.rolling(window=fastk_period).min()
    highest_high = high_series.rolling(window=fastk_period).max()
    
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    k_percent_smooth = k_percent.rolling(window=slowk_period).mean()
    d_percent = k_percent_smooth.rolling(window=slowd_period).mean()
    
    return k_percent_smooth.values, d_percent.values

def ATR(high_prices, low_prices, close_prices, timeperiod=14):
    """Average True Range"""
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=timeperiod).mean()
    
    return atr

def ADX(high_prices, low_prices, close_prices, timeperiod=14):
    """Average Directional Index (simplified)"""
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)
    
    # Calculate directional movement
    plus_dm = high_series.diff()
    minus_dm = low_series.diff() * -1
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed averages
    atr = true_range.rolling(window=timeperiod).mean()
    plus_di = 100 * (plus_dm.rolling(window=timeperiod).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=timeperiod).mean() / atr)
    
    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=timeperiod).mean()
    
    return adx.values
