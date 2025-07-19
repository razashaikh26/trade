import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set up logger
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_trade(self, timestamp, symbol, rsi_value, signal, price, quantity, stop_loss=None, take_profit=None):
        """Log trade information"""
        trade_info = f"TRADE: {symbol} | RSI: {rsi_value:.2f} | Signal: {signal} | Price: {price} | Quantity: {quantity}"
        if stop_loss:
            trade_info += f" | SL: {stop_loss}"
        if take_profit:
            trade_info += f" | TP: {take_profit}"
        
        self.info(trade_info)