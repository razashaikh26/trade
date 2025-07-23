#!/usr/bin/env python3
"""
Email Configuration Tester for CoinDCX Trading Bot
Tests all email notification types to verify configuration
"""

import sys
import os
from datetime import datetime

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from email_notifier import EmailNotifier
    import config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the coindcx directory")
    sys.exit(1)

def test_email_configuration():
    """Test email configuration and connectivity"""
    print("🧪 CoinDCX Trading Bot - Email Configuration Tester")
    print("=" * 60)
    
    # Display current configuration
    print(f"📧 Email Configuration:")
    print(f"   SMTP Server: {config.SMTP_SERVER}:{config.SMTP_PORT}")
    print(f"   Use TLS: {config.EMAIL_USE_TLS}")
    print(f"   Sender: {config.EMAIL_SENDER}")
    print(f"   Receiver: {config.EMAIL_RECEIVER}")
    print(f"   Subject Prefix: {config.EMAIL_SUBJECT_PREFIX}")
    print()
    
    # Initialize email notifier
    try:
        email_notifier = EmailNotifier()
        print("✅ Email notifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize email notifier: {e}")
        return False
    
    return email_notifier

def test_basic_email(email_notifier):
    """Test basic email sending"""
    print("\n🔸 Testing Basic Email...")
    
    try:
        subject = "Test Email from CoinDCX Bot"
        body = f"""
🤖 CoinDCX Trading Bot Email Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Test Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
🎯 Test Type: Basic Email Connectivity

✅ If you receive this email, your email configuration is working correctly!

📧 Configuration Details:
   • SMTP Server: {config.SMTP_SERVER}:{config.SMTP_PORT}
   • Sender: {config.EMAIL_SENDER}
   • TLS Enabled: {config.EMAIL_USE_TLS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Automated test message from CoinDCX Trading Bot
"""
        
        email_notifier._send_email(subject, body)
        print("✅ Basic email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send basic email: {e}")
        return False

def test_trading_notifications(email_notifier):
    """Test all trading notification types"""
    print("\n🔸 Testing Trading Notifications...")
    
    test_symbol = "DOGEINR"
    test_price = 21.5678
    test_quantity = 100.0
    test_order_id = "TEST_ORDER_12345"
    
    notifications_to_test = [
        {
            'type': 'buy',
            'method': 'send_buy_notification',
            'params': {
                'symbol': test_symbol,
                'price': test_price,
                'quantity': test_quantity,
                'order_id': test_order_id,
                'stop_loss': test_price * 0.98,
                'take_profit': test_price * 1.035,
                'confidence': 75.5,
                'analysis': 'RSI oversold + SMA crossover + volume spike confirmation'
            }
        },
        {
            'type': 'sell',
            'method': 'send_sell_notification',
            'params': {
                'symbol': test_symbol,
                'price': test_price * 1.02,
                'quantity': test_quantity,
                'order_id': test_order_id,
                'stop_loss': None,
                'take_profit': None,
                'confidence': 68.2,
                'analysis': 'RSI overbought + resistance rejection + profit target reached'
            }
        },
        {
            'type': 'stop_loss',
            'method': 'send_stop_loss_notification',
            'params': {
                'symbol': test_symbol,
                'price': test_price * 0.98,
                'quantity': test_quantity,
                'order_id': test_order_id,
                'profit_loss': -43.12
            }
        },
        {
            'type': 'take_profit',
            'method': 'send_take_profit_notification',
            'params': {
                'symbol': test_symbol,
                'price': test_price * 1.035,
                'quantity': test_quantity,
                'order_id': test_order_id,
                'profit_loss': 75.47
            }
        },
        {
            'type': 'signal',
            'method': 'send_signal_notification',
            'params': {
                'symbol': test_symbol,
                'signal': 'BUY',
                'confidence': 82.3,
                'analysis': 'Strong bullish confluence: SMC structure shift + RSI divergence + volume breakout',
                'price': test_price
            }
        },
        {
            'type': 'error',
            'method': 'send_error_notification',
            'params': {
                'symbol': test_symbol,
                'error_msg': 'Test error: Insufficient balance for trade execution',
                'action_attempted': 'buy_order'
            }
        }
    ]
    
    success_count = 0
    total_count = len(notifications_to_test)
    
    for notification in notifications_to_test:
        try:
            print(f"   📨 Testing {notification['type']} notification...")
            method = getattr(email_notifier, notification['method'])
            method(**notification['params'])
            print(f"   ✅ {notification['type']} notification sent successfully")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to send {notification['type']} notification: {e}")
    
    print(f"\n📊 Notification Test Results: {success_count}/{total_count} successful")
    return success_count == total_count

def main():
    """Main test function"""
    print("Starting email configuration tests...\n")
    
    # Test 1: Initialize email notifier
    email_notifier = test_email_configuration()
    if not email_notifier:
        print("\n❌ Email configuration test failed. Please check your settings.")
        return
    
    # Test 2: Basic email connectivity
    basic_test_passed = test_basic_email(email_notifier)
    
    if not basic_test_passed:
        print("\n❌ Basic email test failed. Please check your SMTP settings.")
        return
    
    # Test 3: Trading notifications
    print("\n" + "="*60)
    trading_test_passed = test_trading_notifications(email_notifier)
    
    # Final results
    print("\n" + "="*60)
    print("🏁 EMAIL TEST SUMMARY")
    print("="*60)
    
    if basic_test_passed and trading_test_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your email configuration is working correctly")
        print("✅ All notification types are functional")
        print("\n💡 Your trading bot is ready to send email notifications!")
    else:
        print("⚠️  SOME TESTS FAILED")
        if basic_test_passed:
            print("✅ Basic email connectivity works")
        else:
            print("❌ Basic email connectivity failed")
        
        if trading_test_passed:
            print("✅ Trading notifications work")
        else:
            print("❌ Some trading notifications failed")
    
    print("\n📧 Check your email inbox for test messages!")

if __name__ == "__main__":
    main()
