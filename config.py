# Trading configuration settings

# The symbols for the trading pairs to scan
# The bot will trade the first symbol that provides a valid signal
SYMBOLS = ['DOGEUSDT']

# The maximum leverage to use for futures trading
MAX_LEVERAGE = 10

# The percentage of your account balance to risk on a single trade
RISK_PERCENT = 1.0  # e.g., 1.0 means 1% of your account balance

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 2.5  # Adjusted for DOGE volatility

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 1.2  # Adjusted for DOGE price movements

# --- FILTERS ---
# Minimum volatility (ATR as a percentage of price) to consider a trade
MIN_ATR_PERCENT = 0.5  # e.g., 0.5% - avoids flat markets

# Minimum volume (compared to its moving average) to consider a trade
MIN_VOLUME_RATIO = 0.8 # e.g., 0.8 - current volume must be at least 80% of the average

# STRATEGY SETTINGS
# -- Moving Average --
MA_PERIOD = 50           # Period for the trend-following moving average

# -- RSI --
RSI_TIMEFRAME = '15m'  # 15-minute candles
RSI_LENGTH = 14        # A more standard RSI period for smoother signals
RSI_OVERSOLD = 30      # Standard oversold threshold
RSI_OVERBOUGHT = 70    # Standard overbought threshold

# Execution settings
CHECK_INTERVAL = 5     # Check every 5 minutes

# Risk management - OPTIMIZED FOR DOGE
MAX_TRADES_PER_DAY = 6   # Slightly more trades for DOGE
MAX_DAILY_LOSS = 15     # $15 max daily loss for DOGE
MIN_BALANCE = 1       # Minimum account balance in USD