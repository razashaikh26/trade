# SHIB Trading Bot

An advanced cryptocurrency trading bot for SHIB/USDT using Smart Money Concepts (SMC) strategy with comprehensive risk management.

## Features

- **Smart Money Concepts (SMC) Strategy**: Advanced market structure analysis
- **RSI Integration**: Technical indicator for entry/exit signals  
- **Comprehensive Risk Management**: Stop-loss, take-profit, daily limits
- **Multiple Trading Modes**: Live, Mock, and Testnet support
- **Real-time Logging**: Detailed trade and system logs
- **Web Interface**: Flask server for monitoring
- **Binance Futures Integration**: Professional trading platform

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with your Binance API credentials:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

**Security Note**: Never commit your `.env` file to version control.

### 3. Configuration

Edit `config.py` to customize:
- Trading symbol (default: 1000SHIBUSDT)
- Risk parameters
- Strategy settings
- Position sizing

## Usage

### Live Trading
```bash
python main.py
```

### Testnet Trading
```bash
python main.py --testnet
```

### Mock Trading (Simulation)
```bash
python main.py --mock
```

## Risk Management

- **Max Daily Loss**: $10 (configurable)
- **Max Trades/Day**: 5 trades
- **Stop Loss**: 1.5%
- **Take Profit**: 3.0%
- **Minimum Balance**: $5

## Strategy Details

The bot uses Smart Money Concepts combined with RSI:
- **Timeframe**: 15-minute candles
- **RSI Period**: 14
- **Oversold**: 30
- **Overbought**: 70
- **Moving Average**: 50-period trend filter

## Monitoring

The bot includes a Flask web server for health monitoring and deployment compatibility.

## Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves significant risk. Never trade with money you cannot afford to lose.