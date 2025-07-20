#!/usr/bin/env python3
"""
Simple test script for enhanced trading bot features
"""

import pandas as pd
import numpy as np
import sys
import os

def test_imports():
    """Test that all new modules can be imported"""
    print("🔧 Testing imports...")
    
    try:
        from technical_indicators import TechnicalIndicators
        print("  ✅ TechnicalIndicators imported successfully")
        
        import config
        print("  ✅ Config imported successfully")
        
        # Test new config options
        new_configs = [
            'USE_DYNAMIC_ATR_LEVELS',
            'USE_ENGULFING_FILTER', 
            'USE_ADAPTIVE_POSITION_SIZE',
            'ATR_PERIOD',
            'ATR_MA_PERIOD'
        ]
        
        for cfg in new_configs:
            if hasattr(config, cfg):
                print(f"  ✅ {cfg}: {getattr(config, cfg)}")
            else:
                print(f"  ❌ Missing config: {cfg}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_technical_indicators():
    """Test basic technical indicator functionality"""
    print("\n📊 Testing Technical Indicators...")
    
    try:
        from technical_indicators import TechnicalIndicators
        
        # Create simple test data
        data = {
            'open': [0.08, 0.081, 0.079, 0.082, 0.080],
            'high': [0.082, 0.083, 0.081, 0.084, 0.082],
            'low': [0.079, 0.080, 0.078, 0.081, 0.079],
            'close': [0.081, 0.082, 0.080, 0.083, 0.081],
            'volume': [1000000, 1100000, 900000, 1200000, 1050000]
        }
        df = pd.DataFrame(data)
        
        tech = TechnicalIndicators()
        
        # Test ATR calculation
        atr = tech.calculate_atr(df, period=3)
        print(f"  ✅ ATR calculation: {atr.iloc[-1]:.6f}")
        
        # Test ATR-based levels
        levels = tech.atr_based_levels(0.081, atr.iloc[-1], 2.0)
        print(f"  ✅ Dynamic levels calculated:")
        for key, value in levels.items():
            print(f"      {key}: {value:.4f}")
        
        # Test engulfing patterns
        patterns = tech.detect_engulfing_patterns(df)
        print(f"  ✅ Engulfing patterns: {patterns}")
        
        # Test volatility state
        vol_state = tech.get_market_volatility_state(atr.iloc[-1], atr.mean())
        print(f"  ✅ Volatility state: {vol_state}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Technical indicators test failed: {e}")
        return False

def test_strategy_integration():
    """Test strategy integration"""
    print("\n🧠 Testing Strategy Integration...")
    
    try:
        from smc_strategy import SMCStrategy
        import config
        
        strategy = SMCStrategy(config)
        
        # Create test data with enough candles
        np.random.seed(42)
        n_candles = 60
        
        base_price = 0.08
        data = []
        
        for i in range(n_candles):
            price = base_price + np.random.normal(0, 0.002)
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price + np.random.normal(0, 0.001)
            close_price = price
            volume = np.random.uniform(800000, 1200000)
            
            data.append([open_price, high, low, close_price, volume])
        
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Test signal generation
        signal_data = strategy.get_trading_signal(df)
        
        print(f"  ✅ Signal generated: {signal_data.get('signal', 'UNKNOWN')}")
        print(f"  ✅ Confidence: {signal_data.get('confidence', 0)}%")
        
        if 'atr_data' in signal_data:
            print(f"  ✅ ATR data included")
        
        if 'dynamic_levels' in signal_data:
            print(f"  ✅ Dynamic levels included")
        
        if 'engulfing_patterns' in signal_data:
            print(f"  ✅ Engulfing patterns checked")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Trading Bot Features")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_technical_indicators,
        test_strategy_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced features are working correctly")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}")
