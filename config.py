# Trading configuration settings

# The symbol for the trading pair (e.g., 'DOGEUSDT')
# Binance uses DOGEUSDT for DOGE spot/futures trading
SYMBOL = 'DOGEUSDT'  # Changed to DOGE trading pair

# The leverage to use for futures trading
LEVERAGE = 1

# The amount of the asset to trade in each order
# DOGE trading - adjusted for DOGEUSDT format
LOT_SIZE = 50  # About $5-10 per trade (DOGE price range)

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 2.5  # Adjusted for DOGE volatility

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 1.2  # Adjusted for DOGE price movements


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
MIN_BALANCE = 10       # Minimum account balance in USD