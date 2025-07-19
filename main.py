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
    binance, strategy, risk_manager, logger = initialize(mock_mode, testnet)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Running trade check at {current_time}")

    try:
        # First, check if a position is already open for ANY of the symbols.
        # This simple logic prevents opening more than one position at a time.
        an_open_position_exists = False
        for symbol in config.SYMBOLS:
            open_positions = binance.get_open_positions(symbol)
            if open_positions:
                position = open_positions[0]
                entry_price = float(position['entryPrice'])
                contracts = float(position['contracts'])
                position_side = 'buy' if contracts > 0 else 'sell'
                logger.info(f"An open {position_side} position for {symbol} is being managed on the exchange.")
                logger.info(f"Entry: {entry_price}, Size: {abs(contracts)}. Bot will not take new action.")
                an_open_position_exists = True
                break  # Exit the loop as soon as we find one open position

        if an_open_position_exists:
            return # Stop the function if a trade is already active

        # --- If no positions are open, look for a new trading opportunity ---
        logger.info("No open positions found. Scanning for new trading opportunities...")
        account_balance = binance.get_account_balance('USDT')
        
        # The risk manager now needs the binance client to check PNL
        can_trade, reason = risk_manager.can_trade(account_balance, binance, config.SYMBOLS)
        if not can_trade:
            logger.warning(f"Risk manager check failed: {reason}. Halting trade.")
            return
        logger.info(f"Account balance: {account_balance} USDT")

        # Iterate through each symbol to find the first valid signal
        for symbol in config.SYMBOLS:
            logger.info(f"--- Checking {symbol} ---")
            
            klines = binance.get_klines(symbol, '15m')  # Use a timeframe suitable for SMC
            if klines.empty or len(klines) < 50:
                logger.warning(f"Could not retrieve sufficient klines for {symbol}.")
                continue # Move to the next symbol

            # --- CANDLE CLOSE CONFIRMATION ---
            # The last row of klines is the current, unclosed candle. We drop it to only analyze closed candles.
            closed_klines = klines.iloc[:-1]

            signal, analysis = strategy.generate_smc_signal(closed_klines)
            logger.info(f"SMC Signal for {symbol} on closed candles: {signal}")

            if signal in ['BUY', 'SELL']:
                # To get the most up-to-date price for order execution, we use the original klines
                analysis['current_price'] = klines['close'].iloc[-1]

                trade_setup = strategy.calculate_risk_reward_levels(signal, analysis)
                if not trade_setup:
                    logger.warning(f"Could not calculate a valid trade setup for {symbol}.")
                    continue

                entry_price = trade_setup['entry']
                stop_loss = trade_setup['stop_loss']
                take_profit = trade_setup['take_profit']
                rr_ratio = trade_setup.get('rr_ratio', 0)

                logger.info(f"High-Confluence {signal} Setup Found for {symbol}!")
                logger.info(f"  Entry: {entry_price:.5f}")
                logger.info(f"  Stop Loss: {stop_loss:.5f}")
                logger.info(f"  Take Profit: {take_profit:.5f}")
                logger.info(f"  Risk/Reward Ratio: {rr_ratio:.2f}:1")

                if rr_ratio < 5:
                    logger.warning(f"Trade setup for {symbol} ignored. RR ratio {rr_ratio:.2f}:1 is below the 5:1 minimum.")
                    continue

                # --- DYNAMIC POSITION SIZING ---
                risk_amount_usd = account_balance * (config.RISK_PERCENT / 100)
                sl_distance = abs(entry_price - stop_loss)

                if sl_distance == 0:
                    logger.warning("Stop loss distance is zero. Cannot calculate position size.")
                    continue

                quantity = risk_amount_usd / sl_distance
                
                # TODO: Get symbol-specific precision for rounding quantity
                quantity = round(quantity, 3 if 'BTC' in symbol else 0) # Simple rounding rule

                if quantity == 0:
                    logger.warning(f"Calculated quantity ({quantity}) is too small to trade. Skipping trade.")
                    continue

                logger.info(f"Calculated position size: {quantity} {symbol.replace('USDT', '')} (Risking ${risk_amount_usd:.2f})")

                # Place the order with the calculated quantity and SL/TP
                order = binance.place_order(
                    symbol=symbol, 
                    side=signal.lower(), 
                    quantity=quantity, 
                    order_type='market',
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                if order:
                    logger.info(f"Successfully placed new {signal} order for {symbol} with SL/TP.")
                    risk_manager.record_trade() # Now just records the count
                    break # Exit the loop after taking a trade
                else:
                    logger.error(f"Failed to place new order for {symbol}.")
            else:
                logger.info(f"Signal for {symbol} is HOLD. Checking next symbol.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in check_and_trade: {e}")

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