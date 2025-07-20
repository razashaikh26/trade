import time
import schedule
import threading
import argparse
import os
import requests
from flask import Flask
from datetime import datetime
import pandas as pd

# Import project modules
from binance_client import BinanceClient
from smc_strategy import SMCStrategy
from risk_manager import RiskManager
from logger import Logger
from technical_indicators import TechnicalIndicators
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
                
                # Calculate TP and SL based on strategy - Enhanced with ATR
                if getattr(config, 'USE_DYNAMIC_ATR_LEVELS', False):
                    # Get market data for ATR calculation
                    market_data = binance.get_klines(symbol, '15m', limit=100)
                    if market_data is not None and len(market_data) > 0:
                        tech_indicators = TechnicalIndicators()
                        atr = tech_indicators.calculate_atr(market_data, period=config.ATR_PERIOD)
                        
                        if not atr.empty and not pd.isna(atr.iloc[-1]):
                            current_atr = atr.iloc[-1]
                            atr_ma = atr.rolling(window=config.ATR_MA_PERIOD).mean().iloc[-1]
                            volatility_state = tech_indicators.get_market_volatility_state(current_atr, atr_ma)
                            atr_multiplier = tech_indicators.calculate_dynamic_atr_multiplier(volatility_state)
                            
                            # Calculate dynamic levels based on entry price
                            dynamic_levels = tech_indicators.atr_based_levels_for_position(entry_price, current_atr, atr_multiplier)
                            
                            if side.lower() == 'short':
                                take_profit = dynamic_levels['short_take_profit']
                                stop_loss = dynamic_levels['short_stop_loss']
                                logger.info(f"🎯 Dynamic ATR levels (SHORT) - Volatility: {volatility_state}")
                            else:  # long
                                take_profit = dynamic_levels['long_take_profit']
                                stop_loss = dynamic_levels['long_stop_loss']
                                logger.info(f"🎯 Dynamic ATR levels (LONG) - Volatility: {volatility_state}")
                            
                            logger.info(f"ATR: {current_atr:.6f}, Multiplier: {atr_multiplier}x")
                            logger.info(f"📊 Entry-based levels - SL: {stop_loss:.4f}, TP: {take_profit:.4f}")
                        else:
                            # Fallback to fixed percentages
                            logger.warning("⚠️ Could not calculate ATR, using fixed percentages")
                            if side.lower() == 'short':
                                take_profit = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                                stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
                            else:  # long
                                take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                                stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
                    else:
                        # Fallback to fixed percentages
                        logger.warning("⚠️ Could not get market data for ATR, using fixed percentages")
                        if side.lower() == 'short':
                            take_profit = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                            stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
                        else:  # long
                            take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                            stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
                else:
                    # Use fixed percentages (legacy mode)
                    if side.lower() == 'short':
                        take_profit = entry_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                        stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
                    else:  # long
                        take_profit = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                        stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)

                logger.info(f"Current Price: {current_price:.4f}, TP: {take_profit:.4f}, SL: {stop_loss:.4f}")
                
                # Debug: Show exact comparison values
                logger.info(f"🔍 Debug - Entry: {entry_price:.4f}, Current: {current_price:.4f}")
                logger.info(f"🔍 Debug - Side: {side}, TP: {take_profit:.4f}, SL: {stop_loss:.4f}")
                
                # Add tolerance for floating-point precision (0.0001 = 1 pip for most crypto pairs)
                tolerance = 0.0001
                
                # Check if TP or SL should be triggered with tolerance
                if side.lower() == 'short':
                    # For SHORT: TP when price goes DOWN, SL when price goes UP
                    if current_price <= (take_profit + tolerance):
                        logger.info(f"✅ Take Profit hit for SHORT position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} <= {take_profit:.4f} (±{tolerance}) = True")
                        # Close short position by buying back (reduce_only=True)
                        order_result = binance.place_order(symbol, 'BUY', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            profit = (entry_price - current_price) * abs(position_size)
                            logger.info(f"💰 Profit: ${profit:.2f}")
                        else:
                            logger.error("❌ Failed to close SHORT position")
                        return
                    elif current_price >= (stop_loss - tolerance):
                        logger.info(f"🛑 Stop Loss hit for SHORT position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} >= {stop_loss:.4f} (±{tolerance}) = True")
                        # Close short position by buying back (reduce_only=True)
                        order_result = binance.place_order(symbol, 'BUY', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            loss = (current_price - entry_price) * abs(position_size)
                            logger.info(f"💸 Loss: ${loss:.2f}")
                        else:
                            logger.error("❌ Failed to close SHORT position")
                        return
                else:  # long
                    # For LONG: TP when price goes UP, SL when price goes DOWN
                    if current_price >= (take_profit - tolerance):
                        logger.info(f"✅ Take Profit hit for LONG position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} >= {take_profit:.4f} (±{tolerance}) = True")
                        # Close long position by selling (reduce_only=True)
                        order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            profit = (current_price - entry_price) * abs(position_size)
                            logger.info(f"💰 Profit: ${profit:.2f}")
                        else:
                            logger.error("❌ Failed to close LONG position")
                        return
                    elif current_price <= (stop_loss + tolerance):
                        logger.info(f"🛑 Stop Loss hit for LONG position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} <= {stop_loss:.4f} (±{tolerance}) = True")
                        # Close long position by selling (reduce_only=True)
                        order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            loss = (entry_price - current_price) * abs(position_size)
                            logger.info(f"💸 Loss: ${loss:.2f}")
                        else:
                            logger.error("❌ Failed to close LONG position")
                        return
                
                logger.info("Holding position. No action needed.")
                return
                
        except Exception as e:
            logger.error(f"❌ Error checking positions: {e}")
        
        # If no open positions, show balance and look for new trading opportunities
        balance = binance.get_account_balance()
        if binance.mock_mode:
            logger.info(f"Account balance: {balance:.2f} USDT (MOCK DATA)")
        else:
            logger.info(f"Account balance: {balance:.2f} USDT")
        
        # CRITICAL: Double-check no positions exist before opening new ones
        try:
            positions_check = binance.get_open_positions(symbol)
            for pos in positions_check:
                contracts = float(pos.get('contracts', 0))
                if contracts != 0:
                    logger.warning(f"⚠️ SAFETY CHECK: Found existing position with {contracts} contracts. Skipping new position opening.")
                    return
        except Exception as e:
            logger.error(f"❌ Error in safety position check: {e}")
            return
        
        logger.info(f"🔍 Analyzing {symbol} for new trading opportunities...")
        
        # Get market data for analysis
        market_data = binance.get_klines(symbol, '15m', limit=100)
        if market_data is None or len(market_data) == 0:
            logger.error(f"❌ Failed to get market data for {symbol}")
            return
        
        # Get trading signal with enhanced features
        signal_data = strategy.get_trading_signal(market_data)
        signal = signal_data.get("signal", "HOLD")
        confidence = signal_data.get("confidence", 0)
        reason = signal_data.get("reason", "No reason provided")
        
        # Log enhanced signal information
        logger.info(f"📊 Trading Signal: {signal} (Confidence: {confidence}%)")
        logger.info(f"📝 Reason: {reason}")
        
        # Log ATR data if available
        atr_data = signal_data.get('atr_data', {})
        if atr_data:
            logger.info(f"📈 ATR Analysis:")
            logger.info(f"   Current ATR: {atr_data.get('current_atr', 0):.6f}")
            logger.info(f"   ATR %: {atr_data.get('atr_percent', 0):.2f}%")
            logger.info(f"   Volatility: {atr_data.get('volatility_state', 'UNKNOWN')}")
        
        # Log engulfing patterns if available
        engulfing = signal_data.get('engulfing_patterns', {})
        if engulfing:
            if engulfing.get('bullish_engulfing'):
                logger.info("🟢 Bullish Engulfing Pattern Detected")
            if engulfing.get('bearish_engulfing'):
                logger.info("� Bearish Engulfing Pattern Detected")
        
        if signal in ["BUY", "SELL"]:
            # Check risk management
            if not risk_manager.can_trade():
                logger.warning("⚠️ Risk management prevents trading")
                return
            
            # Calculate position size with adaptive sizing if enabled
            balance = binance.get_account_balance()
            
            if getattr(config, 'USE_ADAPTIVE_POSITION_SIZE', False):
                adaptive_data = signal_data.get('adaptive_sizing', {})
                if adaptive_data and adaptive_data.get('current_atr', 0) > 0:
                    tech_indicators = TechnicalIndicators()
                    position_size = tech_indicators.calculate_volatility_adjusted_position_size(
                        account_balance=balance,
                        base_risk_percent=config.RISK_PERCENT,
                        current_atr=adaptive_data['current_atr'],
                        price=current_price,
                        atr_period_avg=adaptive_data.get('atr_ma', adaptive_data['current_atr']),
                        min_size=config.MIN_ORDER_SIZE_DOGE,
                        max_size_multiplier=getattr(config, 'MAX_POSITION_MULTIPLIER', 2.0)
                    )
                    logger.info(f"🎯 Adaptive Position Size: {position_size:.2f} {symbol.replace('USDT', '')}")
                    logger.info(f"   Volatility State: {adaptive_data.get('volatility_state', 'UNKNOWN')}")
                else:
                    # Fallback to standard position sizing
                    position_size = (balance * config.RISK_PERCENT / 100) / current_price
                    logger.info(f"📊 Standard Position Size: {position_size:.2f} {symbol.replace('USDT', '')}")
            else:
                # Standard position sizing
                position_size = (balance * config.RISK_PERCENT / 100) / current_price
                logger.info(f"📊 Standard Position Size: {position_size:.2f} {symbol.replace('USDT', '')}")
            
            # Ensure minimum position size
            if position_size < config.MIN_ORDER_SIZE_DOGE:
                position_size = config.MIN_ORDER_SIZE_DOGE
                logger.info(f"⚠️ Adjusted to minimum position size: {position_size}")
            
            # Calculate stop loss and take profit with dynamic levels
            dynamic_levels = signal_data.get('dynamic_levels', {})
            
            if dynamic_levels and getattr(config, 'USE_DYNAMIC_ATR_LEVELS', False):
                if signal == "BUY":
                    stop_loss = dynamic_levels['long_stop_loss']
                    take_profit = dynamic_levels['long_take_profit']
                    logger.info(f"🎯 Dynamic Levels (LONG): SL={stop_loss:.4f}, TP={take_profit:.4f}")
                else:  # SELL
                    stop_loss = dynamic_levels['short_stop_loss']
                    take_profit = dynamic_levels['short_take_profit']
                    logger.info(f"🎯 Dynamic Levels (SHORT): SL={stop_loss:.4f}, TP={take_profit:.4f}")
            else:
                # Fallback to fixed percentages
                if signal == "BUY":
                    stop_loss = current_price * (1 - config.STOP_LOSS_PERCENT / 100)
                    take_profit = current_price * (1 + config.TAKE_PROFIT_PERCENT / 100)
                    logger.info(f"📊 Fixed Levels (LONG): SL={stop_loss:.4f}, TP={take_profit:.4f}")
                else:  # SELL
                    stop_loss = current_price * (1 + config.STOP_LOSS_PERCENT / 100)
                    take_profit = current_price * (1 - config.TAKE_PROFIT_PERCENT / 100)
                    logger.info(f"📊 Fixed Levels (SHORT): SL={stop_loss:.4f}, TP={take_profit:.4f}")

            if position_size > 0:
                if signal == 'BUY':
                    logger.info(f"� Executing LONG order for {position_size} {symbol} at {current_price:.4f}")
                    order = binance.place_order(symbol, 'BUY', position_size, 'market')
                    if order:
                        logger.info(f"✅ LONG order placed successfully")
                else:  # SELL
                    logger.info(f"🔻 Executing SHORT order for {position_size} {symbol} at {current_price:.4f}")
                    order = binance.place_order(symbol, 'SELL', position_size, 'market')
                    if order:
                        logger.info(f"✅ SHORT order placed successfully")
            else:
                logger.warning("⚠️ Position size calculation error")
        else:
            logger.info("⏸️ No trading signal. Waiting for next opportunity...")
            
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