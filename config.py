# CoinDCX Spot Trading Configuration Settings

# The symbols for the trading pairs to scan (INR pairs for CoinDCX)
# The bot will trade the first symbol that provides a valid signal
SYMBOLS = ['DOGEINR']

# --- TIME-BASED SETTINGS ---
# Timeframe for candlestick data (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 1d)
TIMEFRAME = '15m'  # Changed from '15m' to '1h' for better stability

# Number of candles to fetch for analysis
LOOKBACK_PERIODS = 100  # Number of candles to analyze

# Minimum number of candles required for analysis
MIN_CANDLES_FOR_ANALYSIS = 20  # At least 20 candles needed

# Time between data refreshes in seconds
CHECK_INTERVAL = 300  # 5 minutes

# --- ENHANCED RISK MANAGEMENT ---
# The percentage of your account balance to risk on a single trade
RISK_PERCENT = 2.0  # 1% risk per trade

# Maximum daily risk exposure (% of total balance)
MAX_DAILY_RISK = 5.0  # Maximum 3% of balance at risk per day

# Maximum number of concurrent positions
MAX_CONCURRENT_POSITIONS = 1

# Minimum time between trades (seconds) to avoid overtrading
MIN_TIME_BETWEEN_TRADES = 1800  # 30 minutes

# The execution interval in seconds
BOT_SLEEP_TIME_SECS = 300  # 5 minutes

# --- ENHANCED PROFIT OPTIMIZATION ---
# LEGACY FIXED PERCENTAGES (kept for fallback)
# The percentage for take-profit from the entry price
TAKE_PROFIT_PERCENT = 4.0  # 4% take profit

# The percentage for stop-loss from the entry price
STOP_LOSS_PERCENT = 2.0  # 2% stop loss

# --- ADVANCED ATR-BASED DYNAMIC SETTINGS ---
# Enable ATR-based dynamic stop loss and take profit
USE_DYNAMIC_ATR_LEVELS = True

# ATR calculation period
ATR_PERIOD = 14  # Standard ATR period for better accuracy

# ATR multiplier for stop loss calculation (will be adjusted based on volatility)
ATR_STOP_MULTIPLIER = 1.5  # Reduced from 1.8 to 1.5 for tighter stops

# Risk/Reward ratio for take profit (TP = SL * RISK_REWARD_RATIO)
RISK_REWARD_RATIO = 2.0  # Adjusted for 2% stop loss and 4% take profit

# ATR moving average period for volatility comparison
ATR_MA_PERIOD = 20

# --- ENHANCED SIGNAL FILTERING ---
# Enable candlestick pattern confirmation
USE_ENGULFING_FILTER = True  # Re-enabled for better signal quality

# Enable volatility-adjusted position sizing
USE_ADAPTIVE_POSITION_SIZE = True

# Maximum position size multiplier in low volatility
MAX_POSITION_MULTIPLIER = 1.3  # Reduced from 1.5 to 1.3 for safer sizing

# --- ADVANCED FILTERS ---
# Minimum volatility (ATR as a percentage of price) to consider a trade
MIN_ATR_PERCENT = 0.5  # Increased from 0.3% to 0.5% for better volatility

# Volume settings - Enhanced volume filtering
MIN_VOLUME_RATIO = 0.01  # Increased from 0.005 for better liquidity
VOLUME_MA_PERIOD = 20  # Volume moving average period

# --- TREND CONFIRMATION ---
# Enable trend confirmation using multiple timeframes
USE_TREND_CONFIRMATION = True
TREND_MA_FAST = 9   # Fast moving average
TREND_MA_SLOW = 21  # Slow moving average
TREND_MA_FILTER = 50  # Long-term trend filter

# --- RSI OPTIMIZATION ---
# RSI settings for better entry timing
RSI_PERIOD = 14
RSI_OVERSOLD = 25   # More conservative oversold level
RSI_OVERBOUGHT = 75 # More conservative overbought level
RSI_DIVERGENCE_LOOKBACK = 10  # Look for RSI divergence

# --- MARKET REGIME DETECTION ---
# Enable market regime detection for adaptive strategy
USE_MARKET_REGIME = True
REGIME_LOOKBACK = 50  # Periods to look back for regime detection
TRENDING_THRESHOLD = 0.7  # ADX threshold for trending market
RANGING_THRESHOLD = 0.3   # ADX threshold for ranging market

# --- PROFIT PROTECTION ---
# Enable trailing stop loss
USE_TRAILING_STOP = True
TRAILING_STOP_PERCENT = 0.8  # Trail stop when profit reaches 0.8%
TRAILING_STEP = 0.2  # Move stop by 0.2% increments

# Enable partial profit taking
USE_PARTIAL_PROFITS = True
PARTIAL_PROFIT_LEVELS = [1.5, 2.5]  # Take 25% profit at 1.5%, 25% at 2.5%
PARTIAL_PROFIT_PERCENT = 25  # Percentage to close at each level

# --- DRAWDOWN PROTECTION ---
# Maximum consecutive losses before reducing position size
MAX_CONSECUTIVE_LOSSES = 3
DRAWDOWN_REDUCTION_FACTOR = 0.5  # Reduce position size by 50% after max losses

# Daily loss limit (% of balance)
DAILY_LOSS_LIMIT = 3.0  # Stop trading if daily loss exceeds 3%

# --- ENHANCED TECHNICAL INDICATORS ---
# Bollinger Bands settings
BB_PERIOD = 20
BB_STD_DEV = 2.0

# MACD settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic settings
STOCH_K = 14
STOCH_D = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

# --- EMAIL CONFIGURATION ---
# Email server settings
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_USE_TLS = True

# Email credentials (loaded from environment variables)
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'your_email@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'your_app_password')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER', 'your_email@gmail.com')

# Email notification content
EMAIL_SUBJECT_PREFIX = "[CoinDCX Bot] "

# Execution settings
MAX_TRADES_PER_DAY = 7 
MAX_DAILY_LOSS = 3.0     
MIN_BALANCE = 100    # Minimum balance required in INR

# DOGE-specific settings
MIN_ORDER_SIZE_DOGE = 5
MIN_ORDER_VALUE_INR = 100.0   # Minimum order value in INR

# Position sizing for small accounts
POSITION_SIZE_METHOD = 'fixed_quantity'  
SMALL_ACCOUNT_MODE = True  

# Trading Parameters
SYMBOL = 'DOGEINR'  # Trading pair
TRADE_QUANTITY = 5  # Fixed quantity of 5 DOGE per trade
RISK_PERCENTAGE = 5  # 5% risk per trade (slightly higher for small account)
MIN_BALANCE = 100  # Minimum balance required to trade (INR)
MIN_POSITION_SIZE = 5  # Minimum position size in DOGE (5 DOGE)

# Strategy Parameters
MIN_SIGNAL_CONFIDENCE = 50  # Reduced from 60% to 50% for small accounts

# Risk Management
MAX_DAILY_LOSS_PERCENT = 5  # Increased to 5% for small accounts

# Trading Hours (24-hour format)
TRADING_HOURS = None  # Trade 24/7

# Email Notifications
SEND_EMAIL_ON_ORDER = True
SEND_EMAIL_ON_ERROR = True

# Testing Parameters (overrides for testing)
TEST_MODE = True
TEST_BALANCE = 150  # Set to actual account balance

# Logging
LOG_LEVEL = 'DEBUG'
LOG_TO_FILE = True
LOG_FILE = 'trading_bot.log'

# --- BEGINNER-FRIENDLY STRATEGY FEATURES ---
# Enable individual strategy components
USE_RSI_BUY_LOW_SELL_HIGH = True  # RSI-based contrarian strategy
USE_SMA_EMA_CROSSOVER = True      # Moving average crossover signals
USE_MACD_CROSSOVER = True         # MACD signal line crossover
USE_SUPPORT_RESISTANCE = True     # Support/Resistance bounce strategy
USE_VOLUME_SPIKE_FILTER = True    # Volume spike confirmation

# --- RSI BUY LOW, SELL HIGH STRATEGY ---
RSI_BUY_LOW_THRESHOLD = 30        # Buy when RSI drops below this
RSI_SELL_HIGH_THRESHOLD = 70      # Sell when RSI rises above this
RSI_EXTREME_OVERSOLD = 20         # Strong buy signal
RSI_EXTREME_OVERBOUGHT = 80       # Strong sell signal
RSI_LOOKBACK_PERIODS = 5          # Periods to confirm RSI trend

# --- SMA/EMA CROSSOVER STRATEGY ---
SMA_FAST_PERIOD = 10              # Fast SMA period
SMA_SLOW_PERIOD = 20              # Slow SMA period
EMA_FAST_PERIOD = 12              # Fast EMA period
EMA_SLOW_PERIOD = 26              # Slow EMA period
USE_EMA_OVER_SMA = True           # Prefer EMA over SMA for crossovers
CROSSOVER_CONFIRMATION_PERIODS = 2 # Periods to confirm crossover

# --- MACD CROSSOVER STRATEGY ---
MACD_SIGNAL_CROSSOVER_WEIGHT = 1.5 # Weight for MACD signal crossover
MACD_ZERO_LINE_WEIGHT = 1.0       # Weight for MACD zero line cross
MACD_HISTOGRAM_WEIGHT = 0.5       # Weight for MACD histogram

# --- SUPPORT & RESISTANCE STRATEGY ---
SR_LOOKBACK_PERIODS = 50          # Periods to look back for S/R levels
SR_TOUCH_TOLERANCE = 0.002        # 0.2% tolerance for S/R touch
SR_MIN_TOUCHES = 2                # Minimum touches to confirm S/R level
SR_BOUNCE_CONFIRMATION = 3        # Candles to confirm bounce
SR_STRENGTH_MULTIPLIER = 1.2      # Multiplier for strong S/R levels

# --- VOLUME SPIKE CONFIRMATION ---
VOLUME_SPIKE_MULTIPLIER = 1.5     # Volume must be 1.5x average
VOLUME_SPIKE_LOOKBACK = 20        # Periods for volume average
VOLUME_CONFIRMATION_WEIGHT = 1.3   # Weight boost for volume confirmation
MIN_VOLUME_FOR_TRADE = 100        # Minimum volume required

# --- STRATEGY COMBINATION WEIGHTS ---
# How much each strategy contributes to final signal
STRATEGY_WEIGHTS = {
    'rsi_buy_low_sell_high': 2.0,
    'sma_ema_crossover': 2.5,
    'macd_crossover': 2.0,
    'support_resistance': 1.5,
    'volume_spike': 1.3,
    'smc_analysis': 3.0  # Keep SMC as highest weight
}

# --- EMAIL NOTIFICATIONS FOR ALL ACTIONS ---
# Enhanced email notifications
SEND_EMAIL_ON_BUY = True          # Email when buying
SEND_EMAIL_ON_SELL = True         # Email when selling
SEND_EMAIL_ON_CLOSE = True        # Email when closing position
SEND_EMAIL_ON_STOP_LOSS = True    # Email when stop loss hit
SEND_EMAIL_ON_TAKE_PROFIT = True  # Email when take profit hit
SEND_EMAIL_ON_SIGNAL = False      # Email on every signal (can be spammy)
SEND_EMAIL_ON_ERROR = True        # Email on trading errors

# Email templates for different actions
EMAIL_TEMPLATES = {
    'BUY': '🟢 BUY ORDER EXECUTED',
    'SELL': '🔴 SELL ORDER EXECUTED', 
    'CLOSE': '⚪ POSITION CLOSED',
    'STOP_LOSS': '🛑 STOP LOSS HIT',
    'TAKE_PROFIT': '🎯 TAKE PROFIT HIT',
    'SIGNAL': '📊 TRADING SIGNAL',
    'ERROR': '❌ TRADING ERROR'
}