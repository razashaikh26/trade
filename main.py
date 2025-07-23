import time
import schedule
import threading
import argparse
import os
import requests
from flask import Flask
from datetime import datetime, time
import time as time_module
import logging
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Import project modules
from coindcx_client import CoinDCXClient
from smc_strategy import SMCStrategy
from risk_manager import RiskManager
from logger import Logger
from technical_indicators import TechnicalIndicators
import config
from email_notifier import EmailNotifier

# Initialize logger at module level
logger = logging.getLogger(__name__)

# Global logger instance to prevent multiple instances
_logger_instance = None

def get_logger():
    """Get singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance

def is_market_open():
    """Check if current time is within configured trading hours"""
    try:
        # Get trading hours from config with proper defaults
        trading_hours = getattr(config, 'TRADING_HOURS', {}) or {}
        start_time = trading_hours.get('start', '09:15')
        end_time = trading_hours.get('end', '23:30')
        
        # Parse times with validation
        now = datetime.now().time()
        try:
            start = datetime.strptime(str(start_time), '%H:%M').time()
            end = datetime.strptime(str(end_time), '%H:%M').time()
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid time format in config. Using defaults. Error: {e}")
            start = datetime.strptime('09:15', '%H:%M').time()
            end = datetime.strptime('23:30', '%H:%M').time()
        
        # Check if current time is within trading hours
        if start <= end:
            return start <= now <= end
        else:  # Overnight trading
            return now >= start or now <= end
            
    except Exception as e:
        logger.warning(f"Error checking market hours, defaulting to market open. Error: {e}")
        return True  # Default to True to avoid blocking trades on error

def initialize_components():
    """Initialize all components"""
    logger = get_logger()
    logger.info("🔄 Initializing components...")
    
    # Initialize CoinDCX client
    try:
        coindcx = CoinDCXClient()
        logger.info("✅ CoinDCX client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize CoinDCX client: {e}")
        raise
    
    # Initialize email notifier
    try:
        email_notifier = EmailNotifier()
        logger.info("✅ Email notifier initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize email notifier: {e}")
        email_notifier = None
    
    # Initialize strategy
    try:
        strategy = SMCStrategy(config)
        logger.info("✅ Trading strategy initialized")
    except Exception as e:
        error_msg = f"Failed to initialize trading strategy: {e}"
        logger.error(f"❌ {error_msg}")
        if email_notifier:
            email_notifier.send_error_notification("SYSTEM", error_msg, "initialization")
        raise
    
    # Initialize risk manager
    try:
        risk_manager = RiskManager(
            max_trades_per_day=config.MAX_TRADES_PER_DAY,
            max_daily_loss=config.MAX_DAILY_RISK,
            min_balance=config.MIN_BALANCE
        )
        logger.info("✅ Risk manager initialized")
    except Exception as e:
        error_msg = f"Failed to initialize risk manager: {e}"
        logger.error(f"❌ {error_msg}")
        if email_notifier:
            email_notifier.send_error_notification(symbol, error_msg, "initialization")
        raise
    
    return coindcx, strategy, risk_manager, email_notifier

def check_and_trade():
    """Main trading logic with enhanced features"""
    # Initialize components
    try:
        coindcx, strategy, risk_manager, email_notifier = initialize_components()
        logger = get_logger()  # Get logger instance
        
        logger.info("\n🚀 Starting trading cycle...")
        
        # Get the first symbol from config
        symbol = config.SYMBOLS[0] if config.SYMBOLS else 'DOGEINR'
        logger.info(f"📊 Analyzing {symbol}...")
        
        # Get current price
        current_price = coindcx.get_current_price(symbol)
        if current_price is None:
            error_msg = f"Could not fetch current price for {symbol}"
            logger.error(f"❌ {error_msg}")
            email_notifier.send_error_notification(symbol, error_msg, "price_fetch")
            return
        
        logger.info(f"💰 Current price for {symbol}: ₹{current_price:.6f}")
        
        # Get account balance
        try:
            if getattr(config, 'TEST_MODE', False):
                inr_balance = float(getattr(config, 'TEST_BALANCE', 1000))
                logger.info(f"🔄 TEST MODE - Using test balance: {inr_balance:.2f} INR")
            else:
                # Get all balances and filter for INR
                balances = coindcx.get_account_balance()
                inr_balance = 0.0
                
                if isinstance(balances, list):
                    for balance in balances:
                        if isinstance(balance, dict) and balance.get('currency') == 'INR':
                            inr_balance = float(balance.get('free', 0))
                            break
            
            logger.info(f"💰 Available Balance: {inr_balance:.2f} INR")
            
            if inr_balance < getattr(config, 'MIN_BALANCE', 100):
                logger.warning(f"⚠️ Balance ({inr_balance:.2f} INR) is below minimum required ({getattr(config, 'MIN_BALANCE', 100)} INR)")
                return
                
        except Exception as e:
            logger.error(f"❌ Error fetching account balance: {e}")
            return
        
        # For testing: Simulate 1,000 INR balance
        inr_balance = 1000.0
        logger.info(f"💳 Available INR balance: ₹{inr_balance:.2f}")
        
        # Debug logging for config access
        try:
            logger.debug(f"Checking MIN_BALANCE from config. Available attributes: {[attr for attr in dir(config) if not attr.startswith('__')]}")
            min_balance = config.MIN_BALANCE
            logger.debug(f"MIN_BALANCE value: {min_balance}")
            
            if inr_balance < min_balance:
                logger.warning(f"⚠️ Balance (₹{inr_balance:.2f}) is below minimum required (₹{min_balance:.2f}). Waiting for funds...")
                return
                
        except AttributeError as e:
            logger.error(f"❌ Configuration error: {e}")
            email_notifier.send_error_notification(symbol, f"Configuration error: {e}", "config_error")
            return
            
        # Get candlestick data
        try:
            candles = coindcx.get_candlestick_data(symbol, interval=config.TIMEFRAME, limit=config.LOOKBACK_PERIODS)
            if not candles or len(candles) < config.MIN_CANDLES_FOR_ANALYSIS:
                logger.warning(f"⚠️ Not enough data for analysis. Have {len(candles) if candles else 0} candles, need at least {config.MIN_CANDLES_FOR_ANALYSIS}")
                return
                
            # Convert to DataFrame for analysis
            df = coindcx.format_candlestick_data(candles)
            if df.empty:
                logger.error("❌ Failed to format candlestick data")
                return
                
            logger.info(f"� Fetched {len(df)} candles for analysis")
            
        except Exception as e:
            logger.error(f"❌ Error fetching candlestick data: {e}")
            email_notifier.send_error_notification(symbol, f"Candlestick data error: {e}", "data_fetch")
            return
            
        # Get open orders
        try:
            open_orders = coindcx.get_open_orders(symbol)
            logger.info(f"� Found {len(open_orders)} open orders")
        except Exception as e:
            logger.error(f"❌ Error fetching open orders: {e}")
            email_notifier.send_error_notification(symbol, f"Open orders error: {e}", "orders_fetch")
            return
            
        # Check existing positions and monitor them
        monitor_existing_positions(symbol, current_price, open_orders, email_notifier)
        
        # Generate trading signals with multiple confirmations
        try:
            # Get signals from SMC strategy
            smc_signal, smc_confidence, smc_analysis = strategy.analyze_market(df, symbol)
            
            # Calculate RSI using pandas
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df['rsi'] = calculate_rsi(df['close'])
            current_rsi = df['rsi'].iloc[-1]
            
            # Calculate MACD using pandas
            def calculate_ema(series, period):
                return series.ewm(span=period, adjust=False).mean()
            
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Calculate Bollinger Bands
            def calculate_bollinger_bands(series, window=20, num_std=2):
                sma = series.rolling(window=window).mean()
                std = series.rolling(window=window).std()
                upper_band = sma + (std * num_std)
                lower_band = sma - (std * num_std)
                return upper_band, sma, lower_band
            
            upper_band, middle_band, lower_band = calculate_bollinger_bands(df['close'])
            
            # Calculate current price position relative to Bollinger Bands
            current_price = df['close'].iloc[-1]
            bb_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            # Combine signals with weights
            signals = {
                'smc': (1 if smc_signal == 'BUY' else -1 if smc_signal == 'SELL' else 0) * smc_confidence/100,
                'rsi': -0.5 if current_rsi > 70 else 0.5 if current_rsi < 30 else 0,
                'macd': 1 if df['macd'].iloc[-1] > df['signal_line'].iloc[-1] and df['macd'].iloc[-2] <= df['signal_line'].iloc[-2] else
                       -1 if df['macd'].iloc[-1] < df['signal_line'].iloc[-1] and df['macd'].iloc[-2] >= df['signal_line'].iloc[-2] else 0,
                'bb': 1 if bb_position < 0.2 else -1 if bb_position > 0.8 else 0
            }
            
            # Calculate weighted signal
            weights = {
                'smc': 0.5,  # SMC gets highest weight
                'rsi': 0.2,
                'macd': 0.2,
                'bb': 0.1
            }
            
            weighted_signal = sum(signals[key] * weights[key] for key in signals)
            
            # Determine final signal
            if weighted_signal > 0.3:
                final_signal = 'BUY'
                confidence = min(int(abs(weighted_signal) * 100), 100)
            elif weighted_signal < -0.3:
                final_signal = 'SELL'
                confidence = min(int(abs(weighted_signal) * 100), 100)
            else:
                final_signal = 'HOLD'
                confidence = 0
            
            logger.info(f"📊 Combined Analysis:")
            logger.info(f"   - SMC: {smc_signal} ({smc_confidence:.1f}%)")
            logger.info(f"   - RSI: {current_rsi:.1f} (Oversold <30, Overbought >70)")
            logger.info(f"   - MACD: {'Bullish' if df['macd'].iloc[-1] > df['signal_line'].iloc[-1] else 'Bearish'}")
            logger.info(f"   - BB Position: {bb_position*100:.1f}% (0% = Lower Band, 100% = Upper Band)")
            logger.info(f"🎯 Final Signal: {final_signal} (Confidence: {confidence}%)")
            
            # Check if we're within trading hours and have sufficient balance
            if not is_market_open():
                logger.info("⏳ Market is currently closed. Skipping trade execution.")
                return
                
            # Check if we already have an open position
            open_orders = coindcx.get_open_orders(symbol)
            if open_orders:
                logger.info(f"⏳ Found {len(open_orders)} open orders. Waiting for them to be filled...")
                monitor_existing_positions(symbol, current_price, open_orders, email_notifier)
                return
            
            # Check minimum confidence threshold
            min_confidence = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 60)
            if confidence < min_confidence:
                logger.info(f"⏭️ Signal confidence ({confidence}%) below minimum threshold ({min_confidence}%). Skipping trade.")
                return
            
            # Calculate position size with risk management
            stop_loss_pct = 0.02  # 2% stop loss
            risk_amount = inr_balance * 0.01  # Risk 1% of balance per trade
            position_size = risk_amount / (current_price * stop_loss_pct)
            
            # Ensure minimum position size
            min_position_size = 1.0  # Minimum 1 unit of the asset
            if position_size < min_position_size:
                logger.warning(f"⚠️ Calculated position size ({position_size:.4f}) below minimum ({min_position_size}).")
                return
                
            # Place the order
            order_side = 'buy' if final_signal == 'BUY' else 'sell'
            logger.info(f"📤 Placing {order_side.upper()} order for {position_size:.4f} {symbol} at ₹{current_price:.6f}")
            
            # Uncomment to place real orders
            # order_result = coindcx.place_order(
            #     symbol=symbol,
            #     side=order_side,
            #     order_type='limit',
            #     quantity=position_size,
            #     price=current_price
            # )
            # 
            # if order_result and order_result.get('success'):
            #     logger.info(f"✅ Order placed successfully. ID: {order_result.get('order_id')}")
            #     if email_notifier and getattr(config, 'SEND_EMAIL_ON_ORDER', True):
            #         email_notifier.send_order_notification(
            #             symbol=symbol,
            #             side=order_side.upper(),
            #             price=current_price,
            #             quantity=position_size,
            #             order_id=order_result.get('order_id'),
            #             analysis=f"Signal: {final_signal}, Confidence: {confidence}%"
            #         )
            # else:
            #     logger.error(f"❌ Failed to place order: {order_result.get('message', 'Unknown error')}")
            
        except Exception as e:
            error_msg = f"❌ Error in trading strategy: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if email_notifier and getattr(config, 'SEND_EMAIL_ON_ERROR', True):
                email_notifier.send_error_notification(symbol, error_msg, "strategy_error")
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in trading logic: {e}", exc_info=True)
        if 'email_notifier' in locals() and email_notifier:
            email_notifier.send_error_notification("SYSTEM", f"Unexpected error: {e}", "unexpected_error")
    
    logger.info("✅ Trading cycle completed\n")

def execute_buy_order(symbol, current_price, balance, confidence, analysis):
    """Execute buy order with enhanced notifications"""
    logger = get_logger()
    
    try:
        # Calculate position size with enhanced risk management
        position_size = strategy.calculate_position_size(balance, current_price)
        
        if position_size <= 0:
            logger.warning("⚠️ Position size too small, skipping trade")
            return
        
        # Calculate stop loss and take profit levels
        stop_loss = strategy.calculate_stop_loss(current_price, 'BUY')
        take_profit = strategy.calculate_take_profit(current_price, 'BUY')
        
        logger.info(f"🛒 Executing BUY order:")
        logger.info(f"   💰 Price: ₹{current_price:.6f}")
        logger.info(f"   📊 Quantity: {position_size:.6f}")
        logger.info(f"   💸 Total: ₹{current_price * position_size:.2f}")
        logger.info(f"   🛑 Stop Loss: ₹{stop_loss:.6f}")
        logger.info(f"   🎯 Take Profit: ₹{take_profit:.6f}")
        
        # Place buy order
        order_result = coindcx.place_order(symbol, 'buy', position_size, current_price)
        
        if order_result and order_result.get('success'):
            order_id = order_result.get('order_id')
            logger.info(f"✅ BUY order placed successfully. Order ID: {order_id}")
            
            # Send buy notification
            email_notifier.send_buy_notification(
                symbol=symbol,
                price=current_price,
                quantity=position_size,
                order_id=order_id,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                analysis=analysis
            )
        else:
            error_msg = f"Failed to place BUY order: {order_result.get('error', 'Unknown error')}"
            logger.error(f"❌ {error_msg}")
            email_notifier.send_error_notification(symbol, error_msg, "buy_order")
    
    except Exception as e:
        error_msg = f"Error executing buy order: {str(e)}"
        logger.error(f"❌ {error_msg}")
        email_notifier.send_error_notification(symbol, error_msg, "buy_order")

def execute_sell_order(symbol, current_price, open_orders, confidence, analysis):
    """Execute sell order with enhanced notifications"""
    logger = get_logger()
    
    try:
        for order in open_orders:
            if order.get('side').lower() == 'buy':  # Close buy position
                quantity = float(order.get('quantity', 0))
                buy_price = float(order.get('price', 0))
                
                logger.info(f"🛍️ Executing SELL order:")
                logger.info(f"   💰 Price: ₹{current_price:.6f}")
                logger.info(f"   📊 Quantity: {quantity:.6f}")
                logger.info(f"   💸 Total: ₹{current_price * quantity:.2f}")
                
                # Calculate P&L
                profit_loss = (current_price - buy_price) * quantity
                
                # Place sell order
                order_result = coindcx.place_order(symbol, 'sell', quantity, current_price)
                
                if order_result and order_result.get('success'):
                    order_id = order_result.get('order_id')
                    logger.info(f"✅ SELL order placed successfully. Order ID: {order_id}")
                    logger.info(f"💰 P&L: ₹{profit_loss:.2f}")
                    
                    # Send sell notification
                    email_notifier.send_sell_notification(
                        symbol=symbol,
                        price=current_price,
                        quantity=quantity,
                        order_id=order_id,
                        stop_loss=None,
                        take_profit=None,
                        confidence=confidence,
                        analysis=analysis
                    )
                else:
                    error_msg = f"Failed to place SELL order: {order_result.get('error', 'Unknown error')}"
                    logger.error(f"❌ {error_msg}")
                    email_notifier.send_error_notification(symbol, error_msg, "sell_order")
    
    except Exception as e:
        error_msg = f"Error executing sell order: {str(e)}"
        logger.error(f"❌ {error_msg}")
        email_notifier.send_error_notification(symbol, error_msg, "sell_order")

def monitor_existing_positions(symbol, current_price, open_orders, email_notifier):
    """Monitor existing positions for stop loss and take profit with notifications"""
    logger = get_logger()
    
    if not isinstance(open_orders, list):
        logger.error(f"❌ Invalid open_orders format: {type(open_orders)}")
        return
    
    try:
        for order in open_orders:
            try:
                # Skip if not a dictionary or missing required fields
                if not isinstance(order, dict) or 'side' not in order:
                    logger.warning(f"Skipping invalid order format: {order}")
                    continue
                
                # Only process buy orders for monitoring
                if order.get('side').lower() != 'buy':
                    continue
                    
                # Extract order details with defaults
                order_id = order.get('id', 'unknown')
                buy_price = float(order.get('price', 0))
                quantity = float(order.get('quantity', 0))
                
                if buy_price <= 0 or quantity <= 0:
                    logger.warning(f"Skipping order with invalid price/quantity: {order}")
                    continue
                
                logger.info(f"📊 Monitoring order {order_id} - Price: {buy_price}, Qty: {quantity}")
                
                # Calculate stop loss and take profit levels
                stop_loss = strategy.calculate_stop_loss(buy_price, 'BUY')
                take_profit = strategy.calculate_take_profit(buy_price, 'BUY')
                
                logger.info(f"  - Current: {current_price:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
                
                # Check for stop loss hit
                if current_price <= stop_loss:
                    logger.warning(f"🛑 STOP LOSS HIT for {symbol} at {current_price}")
                    # Handle stop loss logic here
                    
                # Check for take profit hit
                elif current_price >= take_profit:
                    logger.success(f"🎯 TAKE PROFIT HIT for {symbol} at {current_price}")
                    # Handle take profit logic here
                    
            except Exception as e:
                logger.error(f"❌ Error processing order {order.get('id', 'unknown')}: {e}")
                continue
                
    except Exception as e:
        error_msg = f"Error monitoring positions: {str(e)}"
        logger.error(error_msg)
        if email_notifier:
            email_notifier.send_error_notification(
                symbol=symbol,
                message=error_msg,
                error_type="position_monitoring"
            )

def main():
    """Main function to start the trading bot"""
    
    # --- Find and Print Public IP Address ---
    try:
        ip = requests.get('https://api.ipify.org').text
        print(f"\n>>> My public IP address is: {ip}\n>>> Add this IP to your CoinDCX API key whitelist.\n")
    except Exception as e:
        print(f"\nCould not determine public IP address: {e}\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description=f'{config.SYMBOLS[0]} Trading Bot')
    args = parser.parse_args()
    
    # Initialize logger (use singleton)
    logger = get_logger()
    
    # Initialize bot components
    initialize_components()

    # Start the trading bot
    logger.info(f"🤖 Starting {config.SYMBOLS[0]} Trading Bot")
    logger.info(f"💰 Risk per trade: {config.RISK_PERCENT}%")
    logger.info(f"⏰ Check interval: {config.BOT_SLEEP_TIME_SECS} seconds")
    logger.info("✅ BOT IS RUNNING IN LIVE MODE")
    
    # Schedule the trading function
    schedule.every(config.BOT_SLEEP_TIME_SECS).seconds.do(check_and_trade)
    
    # Run the first check immediately
    check_and_trade()
    
    # Keep the bot running with proper scheduling
    logger.info(f"🔄 Bot will check every {config.BOT_SLEEP_TIME_SECS} seconds...")
    while True:
        schedule.run_pending()
        time_module.sleep(10)  # Check every 10 seconds for pending scheduled tasks

# Flask app for keeping the bot alive on cloud platforms
app = Flask(__name__)

@app.route('/')
def home():
    return f"CoinDCX Trading Bot is running! Monitoring {config.SYMBOLS[0]}"

def keep_alive():
    """Keep the bot alive on cloud platforms"""
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    except OSError as e:
        if "Address already in use" in str(e):
            print("⚠️ Port in use, trying alternative port...")
            app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))
        else:
            print(f"❌ Flask server error: {e}")

if __name__ == "__main__":
    # Start the web server in a background thread to keep cloud deployments alive
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    
    # Start the main trading bot
    main()