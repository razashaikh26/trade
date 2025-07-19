import time
import schedule
import threading
import argparse
import os
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
        min_balance=config.MIN_BALANCE,
        stop_loss_percent=config.STOP_LOSS_PERCENT,
        take_profit_percent=config.TAKE_PROFIT_PERCENT
    )
    logger = get_logger()  # Use singleton logger
    
    return binance, strategy, risk_manager, logger

def check_and_trade(mock_mode=False, testnet=False):
    """Main trading logic that runs on schedule"""
    binance, strategy, risk_manager, logger = initialize(mock_mode, testnet)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Running trade check at {current_time}")

    try:
        open_positions = binance.get_open_positions(config.SYMBOL)

        if open_positions:
            # --- MANAGE EXISTING POSITION ---
            position = open_positions[0]
            entry_price = float(position['entryPrice'])
            contracts = float(position['contracts'])
            # Use the 'side' field from the position data, which is more reliable
            position_side = position.get('side', 'buy' if contracts > 0 else 'sell')

            logger.info(f"Managing open {position_side} position for {config.SYMBOL}. Entry: {entry_price}, Size: {abs(contracts)}")

            # Calculate TP and SL prices
            if position_side == 'buy':
                tp_price = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                sl_price = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
            else:  # Short position
                tp_price = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                sl_price = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)

            # Get current market price
            current_price = binance.get_latest_price(config.SYMBOL)
            if not current_price:
                logger.warning("Could not get current price to manage position.")
                return

            logger.info(f"Current Price: {current_price:.2f}, TP: {tp_price:.2f}, SL: {sl_price:.2f}")

            # Check for TP/SL conditions
            close_position = False
            reason = ""
            if position_side == 'buy':
                if current_price >= tp_price:
                    close_position, reason = True, "Take Profit"
                elif current_price <= sl_price:
                    close_position, reason = True, "Stop Loss"
            elif position_side == 'sell':
                if current_price <= tp_price:
                    close_position, reason = True, "Take Profit"
                elif current_price >= sl_price:
                    close_position, reason = True, "Stop Loss"

            if close_position:
                logger.info(f"Closing {position_side} position for {config.SYMBOL} due to {reason}.")
                close_side = 'sell' if position_side == 'buy' else 'buy'
                order = binance.place_order(
                    symbol=config.SYMBOL, side=close_side, quantity=abs(contracts), order_type='market'
                )
                if order:
                    logger.info("Position closed successfully.")
                else:
                    logger.error("Failed to close position.")
            else:
                logger.info("Holding position. No action needed.")

        else:
            # --- LOOK FOR NEW TRADING OPPORTUNITY ---
            logger.info(f"No open position for {config.SYMBOL}. Looking for a new trade.")
            account_balance = binance.get_account_balance('USDT')
            if not risk_manager.can_trade(account_balance):
                logger.warning(f"Risk manager check failed. Halting trade. Balance: {account_balance}")
                return
            logger.info(f"Account balance: {account_balance} USDT")

            klines = binance.get_klines(config.SYMBOL, '15m')  # Use a timeframe suitable for SMC
            if klines.empty or len(klines) < 50:
                logger.warning(f"Could not retrieve sufficient klines for {config.SYMBOL}.")
                return

            signal, analysis = strategy.generate_smc_signal(klines)
            logger.info(f"SMC Signal: {signal}")

            if signal in ['BUY', 'SELL']:
                trade_setup = strategy.calculate_risk_reward_levels(signal, analysis)
                if not trade_setup:
                    logger.warning("Could not calculate a valid trade setup.")
                    return

                entry_price = trade_setup['entry']
                stop_loss = trade_setup['stop_loss']
                take_profit = trade_setup['take_profit']
                rr_ratio = trade_setup.get('rr_ratio', 0)

                logger.info(f"High-Confluence {signal} Setup Found!")
                logger.info(f"  Entry: {entry_price:.2f}")
                logger.info(f"  Stop Loss: {stop_loss:.2f}")
                logger.info(f"  Take Profit: {take_profit:.2f}")
                logger.info(f"  Risk/Reward Ratio: {rr_ratio:.2f}:1")

                # Check if the calculated RR ratio meets the minimum requirement
                if rr_ratio < 5:
                    logger.warning(f"Trade setup ignored. RR ratio {rr_ratio:.2f}:1 is below the 5:1 minimum.")
                    return

                quantity = config.LOT_SIZE
                # NOTE: The following is for a market order. For a real setup,
                # you would likely place a limit order at the entry and set SL/TP.
                order = binance.place_order(
                    symbol=config.SYMBOL, side=signal.lower(), quantity=quantity, order_type='market'
                )
                if order:
                    logger.info(f"Successfully placed new {signal} order for {config.SYMBOL}.")
                    risk_manager.record_trade()
                else:
                    logger.error("Failed to place new order.")
            else:
                logger.info("Signal is HOLD. No action taken.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in check_and_trade: {e}")

def main():
    """Main function to start the trading bot"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=f'{config.SYMBOL} Trading Bot')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode with simulated balance')
    parser.add_argument('--testnet', action='store_true', help='Run in testnet mode')
    args = parser.parse_args()
    
    # Initialize logger (use singleton)
    logger = get_logger()
    
    # Default to testnet mode for real data without risk
    if args.mock:
        logger.info(f"Starting {config.SYMBOL} trading bot in MOCK mode")
    elif args.testnet or not any([args.mock]):  # Default to testnet if no flags
        logger.info(f"Starting {config.SYMBOL} trading bot in TESTNET mode (Real Data)")
        args.testnet = True  # Ensure testnet is set
    else:
        logger.info(f"Starting {config.SYMBOL} trading bot in LIVE mode")
    
    # Run once at startup
    check_and_trade(mock_mode=args.mock, testnet=args.testnet)
    
    # Schedule to run every BOT_SLEEP_TIME_SECS seconds
    if args.mock:
        schedule.every(config.BOT_SLEEP_TIME_SECS).seconds.do(check_and_trade, mock_mode=True, testnet=False)
    elif args.testnet:
        schedule.every(config.BOT_SLEEP_TIME_SECS).seconds.do(check_and_trade, mock_mode=False, testnet=True)
    else:
        schedule.every(config.BOT_SLEEP_TIME_SECS).seconds.do(check_and_trade, mock_mode=False, testnet=False)
    
    # Keep the script running
    logger.info(f"Bot scheduled to run every {config.BOT_SLEEP_TIME_SECS} seconds")
    while True:
        schedule.run_pending()
        time.sleep(1)

# Flask web server to keep Render instance alive
app = Flask(__name__)

@app.route('/')
def hello():
    return "Bot is alive!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    # Start the web server in a background thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    main()