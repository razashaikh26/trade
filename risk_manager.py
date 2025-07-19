from datetime import datetime, time

class RiskManager:
    def __init__(self, max_trades_per_day, max_daily_loss, min_balance, stop_loss_percent, take_profit_percent):
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss = max_daily_loss
        self.min_balance = min_balance
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        
        # Initialize daily tracking
        self.reset_daily_stats()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
    
    def check_new_day(self):
        """Check if it's a new day and reset stats if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily_stats()
            return True
        return False
    
    def can_trade(self, account_balance, trade_amount=0):
        """Check if trading is allowed based on risk parameters"""
        # Check if it's a new day
        self.check_new_day()
        
        # Check account balance
        if account_balance < self.min_balance:
            return False, "Account balance below minimum"
        
        # Check max trades per day
        if self.daily_trade_count >= self.max_trades_per_day:
            return False, "Maximum daily trades reached"
        
        # Check max daily loss
        if self.daily_loss >= self.max_daily_loss:
            return False, "Maximum daily loss reached"
        
        # All checks passed
        return True, "Trade allowed"
    
    def record_trade(self, is_profitable=None, profit_loss=0.0):
        """Record a trade and update statistics"""
        self.daily_trade_count += 1
        
        if is_profitable is False and profit_loss < 0:
            self.daily_loss += abs(profit_loss)
        
        return self.daily_trade_count, self.daily_loss
    
    def get_stop_loss(self):
        """Get stop loss percentage"""
        return self.stop_loss_percent
    
    def get_take_profit(self):
        """Get take profit percentage"""
        return self.take_profit_percent