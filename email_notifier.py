import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
import logging

class EmailNotifier:
    """Handles sending email notifications for trade executions"""
    
    @staticmethod
    def send_email(subject, body, is_html=False):
        """
        Send an email notification
        
        Args:
            subject (str): Email subject
            body (str): Email body content
            is_html (bool): Whether the body is HTML formatted
        """
        if not config.ENABLE_EMAIL_NOTIFICATIONS:
            return False
            
        try:
            # Create message container
            msg = MIMEMultipart()
            msg['From'] = config.EMAIL_SENDER
            msg['To'] = config.EMAIL_RECEIVER
            msg['Subject'] = f"{config.EMAIL_SUBJECT_PREFIX}{subject}"
            
            # Attach the body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Create secure connection with server and send email
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                if config.EMAIL_USE_TLS:
                    server.starttls()
                server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                server.send_message(msg)
                
            logging.info(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")
            return False
    
    @staticmethod
    def send_trade_notification(symbol, action, price, quantity, order_id=None, notes=None):
        """
        Send a trade execution notification
        
        Args:
            symbol (str): Trading pair (e.g., 'DOGEUSDT')
            action (str): 'BUY' or 'SELL'
            price (float): Execution price
            quantity (float): Quantity of the asset
            order_id (str, optional): Exchange order ID
            notes (str, optional): Additional notes about the trade
        """
        subject = f"{action} Order Executed - {symbol}"
        
        # Create HTML email body
        html = f"""
        <html>
        <body>
            <h2>Trade Execution Alert</h2>
            <p><strong>Action:</strong> {action}</p>
            <p><strong>Symbol:</strong> {symbol}</p>
            <p><strong>Price:</strong> {price:.8f}</p>
            <p><strong>Quantity:</strong> {quantity:.2f}</p>
            <p><strong>Value:</strong> {price * quantity:.2f} USDT</p>
        """
        
        if order_id:
            html += f"<p><strong>Order ID:</strong> {order_id}</p>"
            
        if notes:
            html += f"<p><strong>Notes:</strong> {notes}</p>"
            
        html += """
            <p>This is an automated notification from your trading bot.</p>
            </body>
        </html>
        """
        
        return EmailNotifier.send_email(subject, html, is_html=True)
