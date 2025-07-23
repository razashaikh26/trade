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
from email_notifier import EmailNotifier

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
    strategy = SMCStrategy(config=config, testnet=testnet)
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
        # Initialize email notifier if enabled
        email_notifier = EmailNotifier() if config.ENABLE_EMAIL_NOTIFICATIONS else None
        
        # Log the start of trade check with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Running trade check at {current_time}")
        
        # Get current market data
        symbol = config.SYMBOLS[0]  # e.g., 'DOGEUSDT'
        
        # Initialize clients if not already done
        if not hasattr(check_and_trade, 'binance_client'):
            check_and_trade.binance_client = BinanceClient(mock_mode=mock_mode, testnet=testnet)
            check_and_trade.strategy = SMCStrategy(config, testnet=testnet)
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
        
        # Check if we have an open position
        positions = binance.get_open_positions(symbol)
        if positions:
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    logger.info(f"✅ OPEN POSITION: {pos}")
                else:
                    logger.info("❌ NO ACTIVE POSITIONS FOUND")
        else:
            logger.info("❌ NO POSITIONS FOUND")
            
        # Check account balance
        balance = binance.get_account_balance()
        logger.info(f"💰 ACCOUNT BALANCE: {balance} USDT")
        
        # Check recent trades
        try:
            trades = binance.get_recent_trades(symbol, limit=5)
            logger.info(f"🔄 LAST 5 TRADES: {trades}")
        except Exception as e:
            logger.error(f"❌ Error fetching recent trades: {e}")
        
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
                # CRITICAL: If there's an open position, ONLY manage it - never look for new trades
                # Get position details from the position info
                position_info = open_position.get('info', {})
                entry_price = float(position_info.get('entryPrice', 0))
                
                # CRITICAL FIX: Use positionAmt from info to determine side correctly
                position_amt = float(position_info.get('positionAmt', 0))
                position_size = abs(position_amt)  # Always use absolute value for size
                
                # Determine side from positionAmt (negative = short, positive = long)
                side = 'short' if position_amt < 0 else 'long'
                
                # Double-check with the parsed 'side' field if available
                parsed_side = open_position.get('side', '').lower()
                if parsed_side in ['long', 'short']:
                    side = parsed_side
                
                # ADDITIONAL SAFETY: Cross-validate position side detection for live trading
                if not testnet and not binance.mock_mode:  # Live trading only
                    if position_amt != 0:
                        detected_side = 'short' if position_amt < 0 else 'long'
                        if side != detected_side:
                            logger.warning(f"⚠️ LIVE TRADING SAFETY: Position side mismatch detected!")
                            logger.warning(f"   Position amount: {position_amt}")
                            logger.warning(f"   Detected from amount: {detected_side}")
                            logger.warning(f"   Parsed side field: {parsed_side}")
                            logger.warning(f"   Using detected side: {detected_side}")
                            side = detected_side  # Use the amount-based detection as primary
                
                # Get current account balance
                balance = binance.get_account_balance()
                
                logger.info(f"Managing open {side.upper()} position for {symbol}. Entry: {entry_price:.4f}, Size: {position_size}")
                logger.info(f"Position amount: {position_amt} (negative=SHORT, positive=LONG)")
                if not testnet and not binance.mock_mode:
                    logger.info(f"🔴 LIVE TRADING MODE - Position side validation: {side.upper()}")
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
                            dynamic_levels = tech_indicators.atr_based_levels_for_position(entry_price, current_atr, atr_multiplier, testnet=testnet)
                            
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
                        logger.info(f"🔄 Attempting to close SHORT position: BUY {abs(position_size)} {symbol}")
                        order_result = binance.place_order(symbol, 'BUY', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            profit = (entry_price - current_price) * abs(position_size)
                            logger.info(f"💰 Profit: ${profit:.2f}")
                            logger.info(f"✅ Position closed successfully. Order ID: {order_result.get('id', 'N/A')}")
                            
                            # Send email notification for closed position
                            if email_notifier:
                                email_notifier.send_trade_notification(
                                    symbol=symbol,
                                    action='CLOSE SHORT',
                                    price=current_price,
                                    quantity=abs(position_size),
                                    order_id=str(order_result.get('id', 'N/A')),
                                    notes=f"Take Profit Hit | Profit: ${profit:.2f}"
                                )
                        else:
                            logger.error("❌ Failed to close SHORT position")
                            logger.error(f"   Entry: {entry_price:.4f}, Current: {current_price:.4f}, Size: {abs(position_size)}")
                            logger.error(f"   This is a critical error - position should have been closed!")
                            
                            # Try alternative closure methods
                            logger.info("🔄 Attempting alternative position closure methods...")
                            
                            # Method 1: Try with smaller quantity (in case of precision issues)
                            try:
                                smaller_qty = round(abs(position_size) * 0.99, 6)  # 99% of position
                                logger.info(f"🔄 Trying with smaller quantity: {smaller_qty}")
                                if testnet:
                                    logger.info(f"🧪 TESTNET: Using limit order for alternative closure")
                                    alt_order = binance.place_order(symbol, 'BUY', smaller_qty, 'limit', price=current_price, reduce_only=True)
                                else:
                                    alt_order = binance.place_order(symbol, 'BUY', smaller_qty, 'market', reduce_only=True)
                                if alt_order:
                                    logger.info(f"✅ Partial closure successful with smaller quantity")
                                else:
                                    logger.error(f"❌ Alternative method 1 failed")
                            except Exception as e:
                                logger.error(f"❌ Alternative method 1 exception: {e}")
                            
                            # Method 2: Check if position still exists
                            try:
                                current_positions = binance.get_open_positions(symbol)
                                if current_positions:
                                    logger.info(f"🔍 Position still exists: {current_positions}")
                                else:
                                    logger.info(f"🔍 No open positions found - may have been closed elsewhere")
                            except Exception as e:
                                logger.error(f"❌ Could not check positions: {e}")
                        
                        # CRITICAL: Always return here - never continue to new trade logic
                        return
                    elif current_price >= (stop_loss - tolerance):
                        logger.info(f"🛑 Stop Loss hit for SHORT position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} >= {stop_loss:.4f} (±{tolerance}) = True")
                        # Close short position by buying back (reduce_only=True)
                        logger.info(f"🔄 Attempting to close SHORT position: BUY {abs(position_size)} {symbol}")
                        order_result = binance.place_order(symbol, 'BUY', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            loss = (current_price - entry_price) * abs(position_size)
                            logger.info(f"💸 Loss: ${loss:.2f}")
                            logger.info(f"✅ Position closed successfully. Order ID: {order_result.get('id', 'N/A')}")
                            
                            # Send email notification for closed position
                            if email_notifier:
                                email_notifier.send_trade_notification(
                                    symbol=symbol,
                                    action='CLOSE SHORT',
                                    price=current_price,
                                    quantity=abs(position_size),
                                    order_id=str(order_result.get('id', 'N/A')),
                                    notes=f"Stop Loss Hit | Loss: ${loss:.2f}"
                                )
                        else:
                            logger.error("❌ Failed to close SHORT position")
                            logger.error(f"   Entry: {entry_price:.4f}, Current: {current_price:.4f}, Size: {abs(position_size)}")
                            logger.error(f"   This is a critical error - position should have been closed!")
                            
                            # Try alternative closure methods
                            logger.info("🔄 Attempting alternative position closure methods...")
                            
                            # Method 1: Try with smaller quantity (in case of precision issues)
                            try:
                                smaller_qty = round(abs(position_size) * 0.99, 6)  # 99% of position
                                logger.info(f"🔄 Trying with smaller quantity: {smaller_qty}")
                                if testnet:
                                    logger.info(f"🧪 TESTNET: Using limit order for alternative closure")
                                    alt_order = binance.place_order(symbol, 'BUY', smaller_qty, 'limit', price=current_price, reduce_only=True)
                                else:
                                    alt_order = binance.place_order(symbol, 'BUY', smaller_qty, 'market', reduce_only=True)
                                if alt_order:
                                    logger.info(f"✅ Partial closure successful with smaller quantity")
                                else:
                                    logger.error(f"❌ Alternative method 1 failed")
                            except Exception as e:
                                logger.error(f"❌ Alternative method 1 exception: {e}")
                            
                            # Method 2: Check if position still exists
                            try:
                                current_positions = binance.get_open_positions(symbol)
                                if current_positions:
                                    logger.info(f"🔍 Position still exists: {current_positions}")
                                else:
                                    logger.info(f"🔍 No open positions found - may have been closed elsewhere")
                            except Exception as e:
                                logger.error(f"❌ Could not check positions: {e}")
                        
                        # CRITICAL: Always return here - never continue to new trade logic
                        return
                else:  # long
                    # For LONG: TP when price goes UP, SL when price goes DOWN
                    if current_price >= (take_profit - tolerance):
                        logger.info(f"✅ Take Profit hit for LONG position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} >= {take_profit:.4f} (±{tolerance}) = True")
                        # Close long position by selling (reduce_only=True)
                        logger.info(f"🔄 Attempting to close LONG position: SELL {abs(position_size)} {symbol}")
                        
                        # Enhanced debugging before order placement
                        logger.info(f"🔍 Pre-order debugging:")
                        logger.info(f"   - Symbol: {symbol}")
                        logger.info(f"   - Position size: {abs(position_size)}")
                        logger.info(f"   - Current price: {current_price:.4f}")
                        logger.info(f"   - Testnet mode: {testnet}")
                        
                        # Use limit orders for testnet to avoid PERCENT_PRICE filter issues
                        if testnet:
                            logger.info(f"🧪 TESTNET: Using limit order at current price to avoid filter restrictions")
                            order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'limit', price=current_price, reduce_only=True)
                        else:
                            order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            profit = (current_price - entry_price) * abs(position_size)
                            logger.info(f"💰 Profit: ${profit:.2f}")
                            logger.info(f"✅ Position closed successfully. Order ID: {order_result.get('id', 'N/A')}")
                            
                            # Send email notification for closed position
                            if email_notifier:
                                email_notifier.send_trade_notification(
                                    symbol=symbol,
                                    action='CLOSE LONG',
                                    price=current_price,
                                    quantity=abs(position_size),
                                    order_id=str(order_result.get('id', 'N/A')),
                                    notes=f"Take Profit Hit | Profit: ${profit:.2f}"
                                )
                        else:
                            logger.error("❌ Failed to close LONG position")
                            logger.error(f"   Entry: {entry_price:.4f}, Current: {current_price:.4f}, Size: {abs(position_size)}")
                            logger.error(f"   This is a critical error - position should have been closed!")
                            
                            # Try alternative closure methods
                            logger.info("🔄 Attempting alternative position closure methods...")
                            
                            # Method 1: Try with smaller quantity (in case of precision issues)
                            try:
                                smaller_qty = round(abs(position_size) * 0.99, 6)  # 99% of position
                                logger.info(f"🔄 Trying with smaller quantity: {smaller_qty}")
                                if testnet:
                                    logger.info(f"🧪 TESTNET: Using limit order for alternative closure")
                                    alt_order = binance.place_order(symbol, 'SELL', smaller_qty, 'limit', price=current_price, reduce_only=True)
                                else:
                                    alt_order = binance.place_order(symbol, 'SELL', smaller_qty, 'market', reduce_only=True)
                                if alt_order:
                                    logger.info(f"✅ Partial closure successful with smaller quantity")
                                else:
                                    logger.error(f"❌ Alternative method 1 failed")
                            except Exception as e:
                                logger.error(f"❌ Alternative method 1 exception: {e}")
                            
                            # Method 2: Check if position still exists
                            try:
                                current_positions = binance.get_open_positions(symbol)
                                if current_positions:
                                    logger.info(f"🔍 Position still exists: {current_positions}")
                                else:
                                    logger.info(f"🔍 No open positions found - may have been closed elsewhere")
                            except Exception as e:
                                logger.error(f"❌ Could not check positions: {e}")
                        
                        # CRITICAL: Always return here - never continue to new trade logic
                        return
                    elif current_price <= (stop_loss + tolerance):
                        logger.info(f"🛑 Stop Loss hit for LONG position! Closing at {current_price:.4f}")
                        logger.info(f"   Condition: {current_price:.4f} <= {stop_loss:.4f} (±{tolerance}) = True")
                        # Close long position by selling (reduce_only=True)
                        logger.info(f"🔄 Attempting to close LONG position: SELL {abs(position_size)} {symbol}")
                        
                        # Enhanced debugging before order placement
                        logger.info(f"🔍 Pre-order debugging:")
                        logger.info(f"   - Symbol: {symbol}")
                        logger.info(f"   - Position size: {abs(position_size)}")
                        logger.info(f"   - Current price: {current_price:.4f}")
                        logger.info(f"   - Testnet mode: {testnet}")
                        
                        # Use limit orders for testnet to avoid PERCENT_PRICE filter issues
                        if testnet:
                            logger.info(f"🧪 TESTNET: Using limit order at current price to avoid filter restrictions")
                            order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'limit', price=current_price, reduce_only=True)
                        else:
                            order_result = binance.place_order(symbol, 'SELL', abs(position_size), 'market', reduce_only=True)
                        if order_result:
                            loss = (entry_price - current_price) * abs(position_size)
                            logger.info(f"💸 Loss: ${loss:.2f}")
                            logger.info(f"✅ Position closed successfully. Order ID: {order_result.get('id', 'N/A')}")
                            
                            # Send email notification for closed position
                            if email_notifier:
                                email_notifier.send_trade_notification(
                                    symbol=symbol,
                                    action='CLOSE LONG',
                                    price=current_price,
                                    quantity=abs(position_size),
                                    order_id=str(order_result.get('id', 'N/A')),
                                    notes=f"Stop Loss Hit | Loss: ${loss:.2f}"
                                )
                        else:
                            logger.error("❌ Failed to close LONG position")
                            logger.error(f"   Entry: {entry_price:.4f}, Current: {current_price:.4f}, Size: {abs(position_size)}")
                            logger.error(f"   This is a critical error - position should have been closed!")
                            
                            # Try alternative closure methods
                            logger.info("🔄 Attempting alternative position closure methods...")
                            
                            # Method 1: Try with smaller quantity (in case of precision issues)
                            try:
                                smaller_qty = round(abs(position_size) * 0.99, 6)  # 99% of position
                                logger.info(f"🔄 Trying with smaller quantity: {smaller_qty}")
                                if testnet:
                                    logger.info(f"🧪 TESTNET: Using limit order for alternative closure")
                                    alt_order = binance.place_order(symbol, 'SELL', smaller_qty, 'limit', price=current_price, reduce_only=True)
                                else:
                                    alt_order = binance.place_order(symbol, 'SELL', smaller_qty, 'market', reduce_only=True)
                                if alt_order:
                                    logger.info(f"✅ Partial closure successful with smaller quantity")
                                else:
                                    logger.error(f"❌ Alternative method 1 failed")
                            except Exception as e:
                                logger.error(f"❌ Alternative method 1 exception: {e}")
                            
                            # Method 2: Check if position still exists
                            try:
                                current_positions = binance.get_open_positions(symbol)
                                if current_positions:
                                    logger.info(f"🔍 Position still exists: {current_positions}")
                                else:
                                    logger.info(f"🔍 No open positions found - may have been closed elsewhere")
                            except Exception as e:
                                logger.error(f"❌ Could not check positions: {e}")
                        
                        # CRITICAL: Always return here - never continue to new trade logic
                        return
                
                logger.info("Holding position. No action needed.")
                # CRITICAL: Always return here when managing an open position
                # This ensures the bot NEVER looks for new trades while a position is open
                return
                
        except Exception as e:
            logger.error(f"❌ Error checking positions: {e}")
            # CRITICAL: If we can't check positions, don't risk opening new ones
            logger.error("Cannot verify position status - skipping all trading to prevent conflicts")
            return
        
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
        reasons = signal_data.get("reasons", ["No reason provided"])
        market_regime = signal_data.get("market_regime", "unknown")
        
        logger.info(f"📊 Trading Signal: {signal} (Confidence: {confidence}%)")
        for reason in reasons:
            logger.info(f"📝 {reason}")
        
        # Skip if no valid signal
        if signal == "HOLD":
            logger.info("⏸️ No trading signal. Waiting for next opportunity...")
            return
            
        # Get current price and account balance
        current_price = float(market_data['close'].iloc[-1])
        balance = binance.get_account_balance()
        
        # Calculate position size using risk management
        risk_per_trade = 1.0  # 1% risk per trade
        
        # Get ATR for dynamic levels
        atr = strategy.tech_indicators.calculate_atr(market_data, period=14).iloc[-1]
        
        # Dynamic risk/reward based on market regime
        risk_reward_ratio = 2.0  # Default
        if market_regime == 'trending':
            risk_reward_ratio = 2.5  # Higher R:R in trending markets
        elif market_regime == 'range':
            risk_reward_ratio = 1.5  # Lower R:R in ranging markets
        
        # Calculate stop loss and take profit
        if signal == "BUY":
            stop_loss = current_price - (atr * 2.0)  # 2 ATR stop
            take_profit = current_price + (atr * 2.0 * risk_reward_ratio)
        else:  # SELL
            stop_loss = current_price + (atr * 2.0)
            take_profit = current_price - (atr * 2.0 * risk_reward_ratio)
        
        # Calculate position size based on risk
        position_size = risk_manager.calculate_position_size(
            account_balance=balance,
            entry_price=current_price,
            stop_loss_price=stop_loss,
            risk_percent=risk_per_trade
        )
        
        # Ensure minimum order size
        min_order_size = 10  # Minimum 10 DOGE
        position_size = max(position_size, min_order_size)
        
        # Log trade details
        logger.info(f"🎯 Trade Setup:")
        logger.info(f"   Market Regime: {market_regime.upper()}")
        logger.info(f"   Entry: {current_price:.6f}")
        logger.info(f"   Stop Loss: {stop_loss:.6f} ({(abs(stop_loss-current_price)/current_price*100):.2f}%)")
        logger.info(f"   Take Profit: {take_profit:.6f} ({(abs(take_profit-current_price)/current_price*100):.2f}%)")
        logger.info(f"   Position Size: {position_size:.2f} {symbol.replace('USDT', '')} (${position_size * current_price:.2f})")
        logger.info(f"   Risk: {risk_per_trade}% of ${balance:.2f} = ${balance * (risk_per_trade/100):.2f}")
        
        # Execute trade
        if signal == "BUY":
            logger.info(f"🚀 Executing BUY order for {position_size:.2f} {symbol} at {current_price:.6f}")
            order = binance.place_order(
                symbol=symbol,
                side='BUY',
                quantity=position_size,
                price=current_price,
                order_type='MARKET'
            )
        else:  # SELL
            logger.info(f"� Executing SELL order for {position_size:.2f} {symbol} at {current_price:.6f}")
            order = binance.place_order(
                symbol=symbol,
                side='SELL',
                quantity=position_size,
                price=current_price,
                order_type='MARKET'
            )
        
        if order and 'orderId' in order:
            logger.info(f"✅ Order executed successfully! Order ID: {order['orderId']}")
            
            # Store trade details for tracking
            trade_details = {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'direction': signal,
                'entry_time': datetime.utcnow().isoformat(),
                'market_regime': market_regime,
                'risk_reward_ratio': risk_reward_ratio
            }
            
            # Optionally save trade to database or file
            self._save_trade_details(trade_details)
            
        else:
            logger.error("❌ Failed to execute order!")
            
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