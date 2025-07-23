# Trading configuration settings

# The symbols for the trading pairs to scan
# The bot will trade the first symbol that provides a valid signal
SYMBOLS = ['DOGEUSDT']

# The maximum leverage to use for futures trading
MAX_LEVERAGE = 5  # Reduced leverage for small account safety

# The percentage of your account balance to risk on a single trade
RISK_PERCENT = 5.0  # Reduced from 8.0 for better risk management

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# LEGACY FIXED PERCENTAGES (kept for fallback)
# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 2.5  # Slightly reduced TP for better win rate

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 1.25  # Tighter SL for better risk management

# --- NEW ATR-BASED DYNAMIC SETTINGS ---
# Enable ATR-based dynamic stop loss and take profit
USE_DYNAMIC_ATR_LEVELS = True

# ATR calculation period
ATR_PERIOD = 10  # Reduced from 14 for more responsive ATR

# ATR multiplier for stop loss calculation (will be adjusted based on volatility)
ATR_STOP_MULTIPLIER = 1.8  # Slightly reduced from 2.0

# Risk/Reward ratio for take profit (TP = SL * RISK_REWARD_RATIO)
RISK_REWARD_RATIO = 1.8  # Increased from 1.5 for better reward/risk

# ATR moving average period for volatility comparison
ATR_MA_PERIOD = 20  # Reduced from 50 for more responsive volatility measurement

# Enable candlestick pattern confirmation
USE_ENGULFING_FILTER = False

# Enable volatility-adjusted position sizing
USE_ADAPTIVE_POSITION_SIZE = True

# Maximum position size multiplier in low volatility
MAX_POSITION_MULTIPLIER = 1.5  # Reduced from 2.0 for better risk control

# --- FILTERS ---
# Minimum volatility (ATR as a percentage of price) to consider a trade
MIN_ATR_PERCENT = 0.3  # Reduced from 0.5% to allow more trades

# Volume settings
MIN_VOLUME_RATIO = 0.005  # Reduced from 0.01 to allow trading in lower volume conditions
VOLUME_MA_PERIOD = 20  # Period for volume moving average

# STRATEGY SETTINGS
# -- Moving Average --
MA_PERIOD = 20  # Reduced from 50 for more trend signals

# -- RSI --
RSI_TIMEFRAME = '15m'  # 15-minute candles
RSI_LENGTH = 10        # Reduced from 14 for more signals
RSI_OVERSOLD = 40      # Increased from 32 to catch more opportunities
RSI_OVERBOUGHT = 60    # Reduced from 68 to catch more opportunities

# Execution settings
CHECK_INTERVAL = 5     # Check every 5 minutes

# Risk management - OPTIMIZED FOR SMALL ACCOUNT (1000 INR ≈ $12 USD)
MAX_TRADES_PER_DAY = 3   # Conservative for small account
MAX_DAILY_LOSS = 2.0     # $2 max daily loss (about 17% of account)
MIN_BALANCE = 8.0        # Minimum $8 balance to keep trading

# DOGE-specific settings
MIN_ORDER_SIZE_DOGE = 20  # Binance minimum order size for DOGE
MIN_ORDER_VALUE_USD = 5.0   # Minimum order value in USD (Binance requirement)

# Position sizing for small accounts
POSITION_SIZE_METHOD = 'fixed_percentage'  # Use percentage-based sizing
SMALL_ACCOUNT_MODE = True  # Enable small account optimizations

# Email Notification Settings
ENABLE_EMAIL_NOTIFICATIONS = True  # Set to False to disable email notifications
EMAIL_RECEIVER = "razavcf@gmail.com,varunloni54@gmail.com"  # Email address to receive notifications
SMTP_SERVER = "smtp.gmail.com"  # SMTP server address
SMTP_PORT = 587  # SMTP port (587 for TLS)
EMAIL_SENDER = "razavcf1@gmail.com"  # Email address sending the notifications
EMAIL_PASSWORD = "azlb ycmn lbha hqcq"  # App password for the sender email
EMAIL_USE_TLS = True  # Use TLS for secure connection

# Email notification content
EMAIL_SUBJECT_PREFIX = "[Trading Bot] "  # Prefix for all email subjects