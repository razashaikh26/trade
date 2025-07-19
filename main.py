import time
import schedule
import threading
import argparse
import os
import requests
from flask import Flask
from datetime import datetime

# Import project modules
from binance_client import BinanceClient
from smc_strategy import SMCStrategy
from risk_manager import RiskManager
from logger import Logger
import config

# Global logger instance to prevent multiple instances
_logger_instance = None

def get_logger():
    """Get singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance

def initialize(mock_mode=False, testnet=False):
    """Initialize the trading bot"""
    # Create instances of required classes
    binance = BinanceClient(mock_mode=mock_mode, testnet=testnet)
    strategy = SMCStrategy(config=config)
    risk_manager = RiskManager(
        max_trades_per_day=config.MAX_TRADES_PER_DAY,
        max_daily_loss=config.MAX_DAILY_LOSS,
        min_balance=config.MIN_BALANCE
    )
    logger = get_logger()  # Use singleton logger
    
    return binance, strategy, risk_manager, logger

def check_and_trade(mock_mode=False, testnet=False):
    """Main trading logic that runs on schedule"""
    logger = get_logger()
    
    try:
        # Log the start of trade check with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Running trade check at {current_time}")
        
        # Get current market data
        symbol = config.SYMBOLS[0]  # e.g., 'DOGEUSDT'
        
        # Initialize clients if not already done
        if not hasattr(check_and_trade, 'binance_client'):
            check_and_trade.binance_client = BinanceClient(mock_mode=mock_mode, testnet=testnet)
            check_and_trade.strategy = SMCStrategy(config)
            check_and_trade.risk_manager = RiskManager(
                max_trades_per_day=getattr(config, 'MAX_TRADES_PER_DAY', 10),
                max_daily_loss=getattr(config, 'MAX_DAILY_LOSS', 100.0),
                min_balance=getattr(config, 'MIN_BALANCE', 50.0)
            )
            
            # Log the actual mode the bot is running in
            if check_and_trade.binance_client.mock_mode:
                logger.warning("⚠️ BOT IS RUNNING IN MOCK MODE - Data is simulated, not real!")
                logger.warning("⚠️ Prices and balances shown are fake test values")
                if check_and_trade.binance_client.connection_failed:
                    logger.error("❌ Binance API connection failed - likely geographic restriction")
                    logger.error("💡 Consider using a VPN or different cloud provider for live trading")
            else:
                logger.info("✅ BOT IS RUNNING IN LIVE MODE - Using real Binance data")
        
        binance = check_and_trade.binance_client
        strategy = check_and_trade.strategy
        risk_manager = check_and_trade.risk_manager
        
        # Get current price
        try:
            current_price = binance.get_latest_price(symbol)
            if binance.mock_mode:
                logger.info(f"Current {symbol} price: {current_price:.4f} (MOCK DATA)")
            else:
                logger.info(f"Current {symbol} price: {current_price:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to get current price for {symbol}: {e}")
            return
        
        # Check for existing positions
        try:
            positions = binance.get_open_positions(symbol)
            open_position = None
            
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts != 0:
                    open_position = pos
                    break

    
    
            if open_position:
                # Get position details from the position info
                position_info = open_position.get('info', {})
                entry_price = float(position_info.get('entryPrice', 0))
                position_size = float(open_position.get('contracts', 0))
                side = 'short' if position_size < 0 else 'long'
                
                # Get current account balance
                balance = binance.get_account_balance()
                
                logger.info(f"Managing open {side} position for {symbol}. Entry: {entry_price:.4f}, Size: {abs(position_size)}")
                logger.info(f"Account balance: {balance:.2f} USDT")
                
                # Calculate TP and SL based on strategy
                if side.lower() == 'short':
                    take_profit = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                    stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
                else:  # long
                    take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                    stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
                
                logger.info(f"Current Price: {current_price:.2f}, TP: {take_profit:.2f}, SL: {stop_loss:.2f}")
                
                # Check if TP or SL should be triggered
                if side.lower() == 'short':
                    if current_price <= take_profit:
                        logger.info(f"✅ Take Profit hit for SHORT position! Closing at {current_price:.4f}")
                        # Close short position by buying back
                        binance.place_order(symbol, 'BUY', abs(position_size), 'market')
                        logger.info(f"💰 Profit: {(entry_price - current_price) * abs(position_size):.2f}")
                        return
                    elif current_price >= stop_loss:
                        logger.info(f"🛑 Stop Loss hit for SHORT position! Closing at {current_price:.4f}")
                        # Close short position by buying back
                        binance.place_order(symbol, 'BUY', abs(position_size), 'market')
                        logger.info(f"💸 Loss: {(current_price - entry_price) * abs(position_size):.2f}")
                        return
                else:  # long
                    if current_price >= take_profit:
                        logger.info(f"✅ Take Profit hit for LONG position! Closing at {current_price:.4f}")
                        # Close long position by selling
                        binance.place_order(symbol, 'SELL', abs(position_size), 'market')
                        logger.info(f"💰 Profit: {(current_price - entry_price) * abs(position_size):.2f}")
                        return
                    elif current_price <= stop_loss:
                        logger.info(f"🛑 Stop Loss hit for LONG position! Closing at {current_price:.4f}")
                        # Close long position by selling
                        binance.place_order(symbol, 'SELL', abs(position_size), 'market')
                        logger.info(f"💸 Loss: {(entry_price - current_price) * abs(position_size):.2f}")
                        return
                
                logger.info("Holding position. No action needed.")
                return
                
        except Exception as e:
            logger.error(f"❌ Error checking positions: {e}")
        
        # If no open positions, show balance and look for new trading opportunities
        balance = binance.get_account_balance()
        logger.info(f"Account balance: {balance:.2f} USDT")
        logger.info(f"🔍 Analyzing {symbol} for new trading opportunities...")
        
        # Get historical data for analysis
        try:
            df = binance.get_klines(symbol, '1h', 100)
            if df is None or df.empty:
                logger.warning(f"⚠️ No historical data available for {symbol}")
                return
                
            logger.info(f"📈 Analyzing {len(df)} candles of historical data")
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical data: {e}")
            return
        
        # Run strategy analysis
        try:
            signal, analysis = strategy.generate_smc_signal(df)
            logger.info(f"🧠 Strategy signal: {signal}")
            
            if signal in ['BUY', 'SELL']:
                # Calculate position size
                balance = binance.get_account_balance()
                logger.info(f"💰 Account balance: {balance:.2f} USDT")
                
                # Calculate position size based on risk percentage
                risk_amount = balance * (config.RISK_PERCENT / 100)
                position_size = risk_amount / current_price
                
                # Round position size to appropriate decimal places
                position_size = round(position_size, 3)
                
                logger.info(f"📊 Calculated position size: {position_size} {symbol} (Risk: {risk_amount:.2f})")
                
                if position_size > 0:
                    if signal == 'BUY':
                        logger.info(f"🚀 Executing LONG order for {symbol} at {current_price:.4f}")
                        order = binance.place_order(symbol, 'BUY', position_size, 'market')
                        if order:
                            logger.info(f"✅ LONG order placed successfully: {order}")
                    else:  # SELL
                        logger.info(f"🔻 Executing SHORT order for {symbol} at {current_price:.4f}")
                        order = binance.place_order(symbol, 'SELL', position_size, 'market')
                        if order:
                            logger.info(f"✅ SHORT order placed successfully: {order}")
                else:
                    logger.warning("⚠️ Position size too small to trade")
            else:
                logger.info("⏸️ No trading signal. Waiting for next opportunity...")
                
        except Exception as e:
            logger.error(f"❌ Strategy analysis failed: {e}")
            
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred in check_and_trade: {e}")
        import traceback
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        
def main():
    """Main function to start the trading bot"""
    # --- Find and Print Public IP Address ---
    try:
        ip = requests.get('https://api.ipify.org').text
        print(f"\n>>> My public IP address is: {ip}\n>>> Add this IP to your Binance API key whitelist.\n")
    except Exception as e:
        print(f"\nCould not determine public IP address: {e}\n")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=f'{config.SYMBOLS[0]} Trading Bot')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode with simulated balance')
    parser.add_argument('--testnet', action='store_true', help='Use Binance Testnet')
    args = parser.parse_args()
    
    # Initialize logger (use singleton)
    logger = get_logger()
    
    # Initialize bot
    binance, strategy, risk_manager, logger = initialize(mock_mode=args.mock, testnet=args.testnet)

    # --- Mode Confirmation ---
    if args.mock:
        logger.info(f"Starting {config.SYMBOLS[0]} trading bot in MOCK mode")
    elif args.testnet:
        logger.info(f"Starting {config.SYMBOLS[0]} trading bot in TESTNET mode")
    else:
        logger.info(f"--- STARTING BOT IN LIVE TRADING MODE ---")

    # Run once at startup, then schedule
    check_and_trade(mock_mode=args.mock, testnet=args.testnet)
    schedule.every(config.BOT_SLEEP_TIME_SECS).seconds.do(check_and_trade, mock_mode=args.mock, testnet=args.testnet)

    logger.info(f"Bot scheduled to run every {config.BOT_SLEEP_TIME_SECS} seconds")

    while True:
        schedule.run_pending()
        time.sleep(1)

def keep_alive():
    app = Flask(__name__)

    @app.route('/')
    def hello():
        return "Bot is alive!"

    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    # Start the web server in a background thread to keep Render alive
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    main()