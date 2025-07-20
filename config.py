# Trading configuration settings

# The symbols for the trading pairs to scan
# The bot will trade the first symbol that provides a valid signal
SYMBOLS = ['DOGEUSDT']

# The maximum leverage to use for futures trading
MAX_LEVERAGE = 5  # Reduced leverage for small account safety

# The percentage of your account balance to risk on a single trade
RISK_PERCENT = 8.0  # Higher percentage needed for small accounts to meet minimum order size

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 3.0  # Slightly higher TP for better risk/reward

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 1.5  # Tighter SL for small account protection

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

# Risk management - OPTIMIZED FOR SMALL ACCOUNT (1000 INR ≈ $12 USD)
MAX_TRADES_PER_DAY = 3   # Conservative for small account
MAX_DAILY_LOSS = 2.0     # $2 max daily loss (about 17% of account)
MIN_BALANCE = 8.0        # Minimum $8 balance to keep trading

# DOGE-specific settings
MIN_ORDER_SIZE_DOGE = 20  # Binance minimum order size for DOGE
MIN_ORDER_VALUE_USD = 5   # Minimum order value in USD (Binance requirement)

# Position sizing for small accounts
POSITION_SIZE_METHOD = 'fixed_percentage'  # Use percentage-based sizing
SMALL_ACCOUNT_MODE = True  # Enable small account optimizations