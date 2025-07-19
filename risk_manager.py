from datetime import datetime, time
from typing import List

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