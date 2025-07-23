from datetime import datetime, time
from typing import List
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_trades_per_day, max_daily_loss, min_balance):
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss = max_daily_loss
        self.min_balance = min_balance
        
        # Initialize daily tracking
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def check_new_day(self):
        """Check if it's a new day and reset stats if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily_stats()
            return True
        return False

    def update_daily_pnl(self, binance_client, symbols: List[str]):
        """Fetch and update the total realized PNL for the day from the exchange."""
        self.check_new_day()
        
        # Get the timestamp for the start of today (UTC)
        today_start_dt = datetime.combine(datetime.utcnow().date(), time.min)
        start_timestamp_ms = int(today_start_dt.timestamp() * 1000)
        
        total_pnl = 0.0
        for symbol in symbols:
            income_history = binance_client.get_income_history(symbol, start_timestamp_ms)
            for income_event in income_history:
                total_pnl += float(income_event.get('income', 0))
        
        self.daily_pnl = total_pnl
    
    def can_trade(self, account_balance, binance_client, symbols: List[str]):
        """Check if trading is allowed based on risk parameters."""
        # 1. Check for a new day and update PNL from the exchange
        self.update_daily_pnl(binance_client, symbols)
        
        # 2. Check account balance
        if account_balance < self.min_balance:
            return False, f"Account balance ({account_balance:.2f}) below minimum ({self.min_balance})"
        
        # 3. Check max trades per day
        if self.daily_trade_count >= self.max_trades_per_day:
            return False, "Maximum daily trades reached"
        
        # 4. Check max daily loss (only if PNL is negative)
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.max_daily_loss:
            return False, f"Maximum daily loss reached (PNL: {self.daily_pnl:.2f})"
        
        return True, "Trade allowed"
    
    def record_trade(self):
        """Record that a trade has been executed."""
        self.daily_trade_count += 1

    def calculate_position_size(self, account_balance, entry_price, stop_loss_price, risk_percent=1.0):
        """
        Calculate position size based on account balance and risk parameters
        
        Args:
            account_balance (float): Current account balance in USDT
            entry_price (float): Entry price of the trade
            stop_loss_price (float): Stop loss price
            risk_percent (float): Percentage of account to risk (default: 1%)
            
        Returns:
            float: Position size in base currency (e.g., DOGE)
        """
        try:
            # Calculate risk amount in USDT
            risk_amount = account_balance * (risk_percent / 100.0)
            
            # Calculate price difference (absolute value)
            price_diff = abs(entry_price - stop_loss_price)
            
            # Handle potential division by zero
            if price_diff == 0:
                return 0
                
            # Calculate position size in base currency
            position_size = (risk_amount / price_diff)
            
            # Apply maximum position size limit (e.g., 50% of account)
            max_position_value = account_balance * 0.5  # Max 50% of account per trade
            max_position_size = max_position_value / entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def calculate_dynamic_sl_tp(self, entry_price, atr, side, risk_reward_ratio=1.5):
        """
        Calculate dynamic stop loss and take profit based on ATR
        
        Args:
            entry_price (float): Entry price
            atr (float): Current ATR value
            side (str): 'long' or 'short'
            risk_reward_ratio (float): Desired risk/reward ratio
            
        Returns:
            tuple: (stop_loss, take_profit)
        """
        # Base ATR multiplier
        atr_multiplier = 2.0  # Default multiplier
        
        # Adjust multiplier based on volatility
        if atr / entry_price < 0.002:  # Low volatility
            atr_multiplier = 2.5
        elif atr / entry_price > 0.005:  # High volatility
            atr_multiplier = 1.5
            
        # Calculate stop loss
        if side.lower() == 'long':
            stop_loss = entry_price - (atr * atr_multiplier)
            take_profit = entry_price + (atr * atr_multiplier * risk_reward_ratio)
        else:  # short
            stop_loss = entry_price + (atr * atr_multiplier)
            take_profit = entry_price - (atr * atr_multiplier * risk_reward_ratio)
            
        # Ensure stop loss is not too tight (min 0.5%)
        min_sl_distance = entry_price * 0.005
        current_sl_distance = abs(entry_price - stop_loss)
        if current_sl_distance < min_sl_distance:
            if side.lower() == 'long':
                stop_loss = entry_price - min_sl_distance
                take_profit = entry_price + (min_sl_distance * risk_reward_ratio)
            else:
                stop_loss = entry_price + min_sl_distance
                take_profit = entry_price - (min_sl_distance * risk_reward_ratio)
                
        return stop_loss, take_profit