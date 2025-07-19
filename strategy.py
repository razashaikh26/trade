import pandas as pd
import pandas_ta as ta

class TradingStrategy:
    """A trading strategy that combines RSI with a Moving Average trend filter."""
    def __init__(self, rsi_length, oversold, overbought, ma_period):
        self.rsi_length = rsi_length
        self.oversold = oversold
        self.overbought = overbought
        self.ma_period = ma_period

    def generate_signal(self, klines):
        """
        Generates a trading signal based on RSI and a Moving Average trend filter.

        :param klines: DataFrame with candlestick data (must have 'close' column)
        :return: A tuple of (signal, rsi_value, ma_value)
        """
        if klines.empty or len(klines) < self.ma_period:
            return 'HOLD', None, None

        # Use a copy to avoid SettingWithCopyWarning
        klines = klines.copy()

        # Calculate indicators
        klines.ta.rsi(length=self.rsi_length, append=True)
        klines.ta.sma(length=self.ma_period, append=True)
        
        # Get the latest values
        last_candle = klines.iloc[-1]
        last_close = last_candle['close']
        last_rsi = last_candle[f'RSI_{self.rsi_length}']
        last_ma = last_candle[f'SMA_{self.ma_period}']

        if pd.isna(last_rsi) or pd.isna(last_ma):
            return 'HOLD', None, None

        # Determine trend
        is_uptrend = last_close > last_ma
        is_downtrend = last_close < last_ma

        # Generate signals with improved logic
        # BUY: RSI oversold (mean reversion) OR (uptrend + RSI recovering from oversold)
        if (last_rsi < self.oversold) or (is_uptrend and last_rsi < 40):
            return 'BUY', last_rsi, last_ma
        
        # SELL: RSI overbought (mean reversion) OR (downtrend + RSI declining from overbought)
        if (last_rsi > self.overbought) or (is_downtrend and last_rsi > 60):
            return 'SELL', last_rsi, last_ma

        return 'HOLD', last_rsi, last_ma