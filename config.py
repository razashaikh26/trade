# Trading configuration settings

# The symbol for the trading pair (e.g., '1000SHIBUSDT')
# Binance uses 1000SHIBUSDT for SHIB futures trading
SYMBOL = '1000SHIBUSDT'  # Correct Binance futures symbol for SHIB

# The leverage to use for futures trading
LEVERAGE = 1

# The amount of the asset to trade in each order
# SHIB trading - adjusted for 1000SHIBUSDT format
LOT_SIZE = 100  # About $1-2 per trade (1000SHIB multiplier)

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 3.0  # Increased to 3% (SHIB is more volatile)

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 1.5  # Increased to 1.5% (SHIB needs wider stops)


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

# Risk management - SAFER SETTINGS FOR SHIB
MAX_TRADES_PER_DAY = 5   # Even more conservative for altcoin
MAX_DAILY_LOSS = 2      # REDUCED to $10 max daily loss (very safe)
MIN_BALANCE = 5        # Minimum account balance in USD (reduced)