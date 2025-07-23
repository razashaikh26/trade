import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import config

class EmailNotifier:
    def __init__(self):
        self.smtp_server = config.SMTP_SERVER
        self.smtp_port = config.SMTP_PORT
        self.sender_email = config.EMAIL_SENDER
        self.sender_password = config.EMAIL_PASSWORD
        self.receiver_emails = config.EMAIL_RECEIVER.split(',')
        self.use_tls = config.EMAIL_USE_TLS
        
    def send_trade_notification(self, action, symbol, price, quantity, order_id=None, 
                              profit_loss=None, stop_loss=None, take_profit=None, 
                              confidence=None, analysis=None, error_msg=None):
        """Send comprehensive trade notification email"""
        try:
            if not self._should_send_email(action):
                return
                
            subject = self._get_email_subject(action, symbol)
            body = self._create_email_body(
                action, symbol, price, quantity, order_id, 
                profit_loss, stop_loss, take_profit, 
                confidence, analysis, error_msg
            )
            
            self._send_email(subject, body)
            print(f"✅ Email notification sent for {action} action")
            
        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")
    
    def send_buy_notification(self, symbol, price, quantity, order_id, stop_loss, take_profit, confidence, analysis):
        """Send buy order notification"""
        self.send_trade_notification(
            action='BUY',
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            analysis=analysis
        )
    
    def send_sell_notification(self, symbol, price, quantity, order_id, stop_loss, take_profit, confidence, analysis):
        """Send sell order notification"""
        self.send_trade_notification(
            action='SELL',
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            analysis=analysis
        )
    
    def send_close_notification(self, symbol, price, quantity, order_id, profit_loss, reason):
        """Send position close notification"""
        self.send_trade_notification(
            action='CLOSE',
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            profit_loss=profit_loss,
            analysis=reason
        )
    
    def send_stop_loss_notification(self, symbol, price, quantity, order_id, profit_loss):
        """Send stop loss hit notification"""
        self.send_trade_notification(
            action='STOP_LOSS',
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            profit_loss=profit_loss
        )
    
    def send_take_profit_notification(self, symbol, price, quantity, order_id, profit_loss):
        """Send take profit hit notification"""
        self.send_trade_notification(
            action='TAKE_PROFIT',
            symbol=symbol,
            price=price,
            quantity=quantity,
            order_id=order_id,
            profit_loss=profit_loss
        )
    
    def send_signal_notification(self, symbol, signal, confidence, analysis, price):
        """Send trading signal notification"""
        if config.SEND_EMAIL_ON_SIGNAL:
            self.send_trade_notification(
                action='SIGNAL',
                symbol=symbol,
                price=price,
                quantity=0,
                confidence=confidence,
                analysis=f"{signal} signal: {analysis}"
            )
    
    def send_error_notification(self, symbol, error_msg, action_attempted=None):
        """Send error notification"""
        if config.SEND_EMAIL_ON_ERROR:
            self.send_trade_notification(
                action='ERROR',
                symbol=symbol,
                price=0,
                quantity=0,
                error_msg=error_msg,
                analysis=f"Error during {action_attempted}" if action_attempted else "Trading error"
            )
    
    def _should_send_email(self, action):
        """Check if email should be sent for this action"""
        email_settings = {
            'BUY': config.SEND_EMAIL_ON_BUY,
            'SELL': config.SEND_EMAIL_ON_SELL,
            'CLOSE': config.SEND_EMAIL_ON_CLOSE,
            'STOP_LOSS': config.SEND_EMAIL_ON_STOP_LOSS,
            'TAKE_PROFIT': config.SEND_EMAIL_ON_TAKE_PROFIT,
            'SIGNAL': config.SEND_EMAIL_ON_SIGNAL,
            'ERROR': config.SEND_EMAIL_ON_ERROR
        }
        return email_settings.get(action, True)
    
    def _get_email_subject(self, action, symbol):
        """Generate email subject based on action"""
        template = config.EMAIL_TEMPLATES.get(action, f' {action}')
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{config.EMAIL_SUBJECT_PREFIX}{template} - {symbol} [{timestamp}]"
    
    def _create_email_body(self, action, symbol, price, quantity, order_id, 
                          profit_loss, stop_loss, take_profit, confidence, analysis, error_msg):
        """Create comprehensive email body"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        body = f"""
 CoinDCX Trading Bot Notification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Time: {timestamp}
 Action: {action}
 Symbol: {symbol}
"""
        
        # Action-specific details
        if action in ['BUY', 'SELL']:
            body += f"""
 Price: ₹{price:.6f}
 Quantity: {quantity:.6f}
 Total Value: ₹{price * quantity:.2f}
 Order ID: {order_id or 'N/A'}
"""
            if confidence:
                body += f" Confidence: {confidence:.1f}%\n"
            if stop_loss:
                body += f" Stop Loss: ₹{stop_loss:.6f}\n"
            if take_profit:
                body += f" Take Profit: ₹{take_profit:.6f}\n"
                
        elif action == 'CLOSE':
            body += f"""
 Close Price: ₹{price:.6f}
 Quantity: {quantity:.6f}
 Total Value: ₹{price * quantity:.2f}
 Order ID: {order_id or 'N/A'}
"""
            if profit_loss is not None:
                pnl_emoji = "" if profit_loss > 0 else ""
                body += f"{pnl_emoji} P&L: ₹{profit_loss:.2f}\n"
                
        elif action in ['STOP_LOSS', 'TAKE_PROFIT']:
            body += f"""
 Exit Price: ₹{price:.6f}
 Quantity: {quantity:.6f}
 Total Value: ₹{price * quantity:.2f}
 Order ID: {order_id or 'N/A'}
"""
            if profit_loss is not None:
                pnl_emoji = "" if profit_loss > 0 else ""
                body += f"{pnl_emoji} P&L: ₹{profit_loss:.2f}\n"
                
        elif action == 'SIGNAL':
            body += f"""
 Current Price: ₹{price:.6f}
 Confidence: {confidence:.1f}%
"""
            
        elif action == 'ERROR':
            body += f"""
 Error: {error_msg}
"""
        
        # Analysis section
        if analysis:
            body += f"""
 Analysis: {analysis}
"""
        
        # Footer
        body += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Automated message from CoinDCX Trading Bot
 Risk per trade: {config.RISK_PERCENT}%
 Strategy: Enhanced SMC with Multi-Indicator Confirmation
"""
        
        return body
    
    def _send_email(self, subject, body):
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.receiver_emails)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f" SMTP error: {e}")
            raise
