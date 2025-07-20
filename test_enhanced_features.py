#!/usr/bin/env python3
"""
Comprehensive test script for enhanced trading bot features
Tests ATR-based dynamic levels, engulfing patterns, and adaptive position sizing
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators
from smc_strategy import SMCStrategy
import config

def create_test_data():
    """Create synthetic market data for testing"""
    print("📊 Creating synthetic test data...")
    
    # Create 100 candles of test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible results
    
    base_price = 0.08  # DOGE price around $0.08
    prices = []
    volumes = []
    
    current_price = base_price
    for i in range(100):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 10)  # Slight sine wave trend
        volatility = np.random.normal(0, 0.002)  # 0.2% volatility
        
        price_change = trend + volatility
        current_price = max(0.01, current_price + price_change)  # Prevent negative prices
        
        # Generate OHLC from current price
        high = current_price * (1 + abs(np.random.normal(0, 0.001)))
        low = current_price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = current_price + np.random.normal(0, 0.0005)
        close_price = current_price
        
        prices.append([open_price, high, low, close_price])
        volumes.append(np.random.uniform(1000000, 5000000))  # Random volume
    
    df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'])
    df['volume'] = volumes
    df['timestamp'] = dates
    
    print(f"✅ Created {len(df)} candles of test data")
    print(f"   Price range: ${df['low'].min():.4f} - ${df['high'].max():.4f}")
    
    return df

def test_technical_indicators():
    """Test the TechnicalIndicators class"""
    print("\n🔧 Testing Technical Indicators...")
    
    df = create_test_data()
    tech = TechnicalIndicators()
    
    # Test ATR calculation
    print("  Testing ATR calculation...")
    atr = tech.calculate_atr(df, period=14)
    assert not atr.empty, "ATR calculation failed"
    assert not pd.isna(atr.iloc[-1]), "ATR contains NaN values"
    print(f"  ✅ ATR calculated successfully. Current ATR: {atr.iloc[-1]:.6f}")
    
    # Test ATR-based levels
    print("  Testing ATR-based levels...")
    current_price = df['close'].iloc[-1]
    current_atr = atr.iloc[-1]
    levels = tech.atr_based_levels(current_price, current_atr, 2.0)
    
    assert 'long_stop_loss' in levels, "Missing long_stop_loss"
    assert 'long_take_profit' in levels, "Missing long_take_profit"
    assert 'short_stop_loss' in levels, "Missing short_stop_loss"
    assert 'short_take_profit' in levels, "Missing short_take_profit"
    
    # Validate level logic
    assert levels['long_stop_loss'] < current_price, "Long SL should be below current price"
    assert levels['long_take_profit'] > current_price, "Long TP should be above current price"
    assert levels['short_stop_loss'] > current_price, "Short SL should be above current price"
    assert levels['short_take_profit'] < current_price, "Short TP should be below current price"
    
    print(f"  ✅ ATR levels: Long SL={levels['long_stop_loss']:.4f}, TP={levels['long_take_profit']:.4f}")
    
    # Test engulfing pattern detection
    print("  Testing engulfing pattern detection...")
    patterns = tech.detect_engulfing_patterns(df)
    assert 'bullish_engulfing' in patterns, "Missing bullish_engulfing"
    assert 'bearish_engulfing' in patterns, "Missing bearish_engulfing"
    print(f"  ✅ Engulfing patterns: Bullish={patterns['bullish_engulfing']}, Bearish={patterns['bearish_engulfing']}")
    
    # Test volatility state
    print("  Testing volatility state detection...")
    atr_ma = atr.rolling(window=20).mean().iloc[-1]
    vol_state = tech.get_market_volatility_state(current_atr, atr_ma)
    assert vol_state in ['LOW', 'NORMAL', 'HIGH'], f"Invalid volatility state: {vol_state}"
    print(f"  ✅ Volatility state: {vol_state}")
    
    # Test adaptive position sizing
    print("  Testing adaptive position sizing...")
    position_size = tech.calculate_volatility_adjusted_position_size(
        account_balance=100.0,
        base_risk_percent=8.0,
        current_atr=current_atr,
        price=current_price,
        atr_period_avg=atr_ma,
        min_size=20.0
    )
    assert position_size >= 20.0, "Position size below minimum"
    print(f"  ✅ Adaptive position size: {position_size:.2f}")
    
    return df

def test_smc_strategy():
    """Test the enhanced SMC strategy"""
    print("\n🧠 Testing Enhanced SMC Strategy...")
    
    df = create_test_data()
    strategy = SMCStrategy(config)
    
    # Test basic signal generation
    print("  Testing signal generation...")
    signal_data = strategy.get_trading_signal(df)
    
    assert 'signal' in signal_data, "Missing signal"
    assert 'confidence' in signal_data, "Missing confidence"
    assert 'reason' in signal_data, "Missing reason"
    assert signal_data['signal'] in ['BUY', 'SELL', 'HOLD'], f"Invalid signal: {signal_data['signal']}"
    
    print(f"  ✅ Signal: {signal_data['signal']} (Confidence: {signal_data['confidence']}%)")
    print(f"      Reason: {signal_data['reason']}")
    
    # Test ATR data inclusion
    if 'atr_data' in signal_data:
        atr_data = signal_data['atr_data']
        print(f"  ✅ ATR Data included:")
        print(f"      Current ATR: {atr_data.get('current_atr', 0):.6f}")
        print(f"      Volatility: {atr_data.get('volatility_state', 'UNKNOWN')}")
    
    # Test dynamic levels
    if 'dynamic_levels' in signal_data and signal_data['dynamic_levels']:
        levels = signal_data['dynamic_levels']
        print(f"  ✅ Dynamic levels calculated:")
        for key, value in levels.items():
            print(f"      {key}: {value:.4f}")
    
    # Test engulfing patterns
    if 'engulfing_patterns' in signal_data:
        patterns = signal_data['engulfing_patterns']
        print(f"  ✅ Engulfing patterns checked:")
        print(f"      Bullish: {patterns.get('bullish_engulfing', False)}")
        print(f"      Bearish: {patterns.get('bearish_engulfing', False)}")
    
    return signal_data

def test_config_flags():
    """Test configuration flag functionality"""
    print("\n⚙️ Testing Configuration Flags...")
    
    # Test that new config options exist
    required_configs = [
        'USE_DYNAMIC_ATR_LEVELS',
        'USE_ENGULFING_FILTER', 
        'USE_ADAPTIVE_POSITION_SIZE',
        'ATR_PERIOD',
        'ATR_MA_PERIOD',
        'MAX_POSITION_MULTIPLIER'
    ]
    
    for config_name in required_configs:
        assert hasattr(config, config_name), f"Missing config: {config_name}"
        value = getattr(config, config_name)
        print(f"  ✅ {config_name}: {value}")
    
    print("  ✅ All configuration flags present")

def test_engulfing_pattern_scenarios():
    """Test specific engulfing pattern scenarios"""
    print("\n🕯️ Testing Engulfing Pattern Scenarios...")
    
    tech = TechnicalIndicators()
    
    # Create bullish engulfing pattern
    bullish_data = pd.DataFrame({
        'open': [0.08, 0.079],      # Previous red, current green
        'high': [0.081, 0.082],
        'low': [0.078, 0.077],      # Current engulfs previous
        'close': [0.079, 0.081],    # Previous red, current green
        'volume': [1000000, 1200000]
    })
    
    patterns = tech.detect_engulfing_patterns(bullish_data)
    assert patterns['bullish_engulfing'] == True, "Failed to detect bullish engulfing"
    assert patterns['bearish_engulfing'] == False, "False positive bearish engulfing"
    print("  ✅ Bullish engulfing pattern detected correctly")
    
    # Create bearish engulfing pattern
    bearish_data = pd.DataFrame({
        'open': [0.079, 0.081],     # Previous green, current red
        'high': [0.081, 0.082],
        'low': [0.078, 0.077],      # Current engulfs previous
        'close': [0.081, 0.078],    # Previous green, current red
        'volume': [1000000, 1200000]
    })
    
    patterns = tech.detect_engulfing_patterns(bearish_data)
    assert patterns['bearish_engulfing'] == True, "Failed to detect bearish engulfing"
    assert patterns['bullish_engulfing'] == False, "False positive bullish engulfing"
    print("  ✅ Bearish engulfing pattern detected correctly")

def test_volatility_scenarios():
    """Test different volatility scenarios"""
    print("\n📈 Testing Volatility Scenarios...")
    
    tech = TechnicalIndicators()
    
    # Test low volatility scenario
    low_vol_atr = 0.001
    normal_atr = 0.002
    vol_state = tech.get_market_volatility_state(low_vol_atr, normal_atr)
    assert vol_state == 'LOW', f"Expected LOW volatility, got {vol_state}"
    print(f"  ✅ Low volatility detected: {vol_state}")
    
    # Test high volatility scenario
    high_vol_atr = 0.005
    vol_state = tech.get_market_volatility_state(high_vol_atr, normal_atr)
    assert vol_state == 'HIGH', f"Expected HIGH volatility, got {vol_state}"
    print(f"  ✅ High volatility detected: {vol_state}")
    
    # Test dynamic multipliers
    low_mult = tech.calculate_dynamic_atr_multiplier('LOW')
    normal_mult = tech.calculate_dynamic_atr_multiplier('NORMAL')
    high_mult = tech.calculate_dynamic_atr_multiplier('HIGH')
    
    assert low_mult < normal_mult < high_mult, "ATR multipliers not in correct order"
    print(f"  ✅ ATR multipliers: LOW={low_mult}, NORMAL={normal_mult}, HIGH={high_mult}")

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Trading Bot Feature Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_data = test_technical_indicators()
        test_smc_strategy()
        test_config_flags()
        test_engulfing_pattern_scenarios()
        test_volatility_scenarios()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ ATR-based dynamic levels working")
        print("✅ Engulfing pattern detection working")
        print("✅ Adaptive position sizing working")
        print("✅ Configuration flags working")
        print("✅ Volatility assessment working")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
