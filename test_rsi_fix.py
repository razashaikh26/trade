#!/usr/bin/env python3
"""
Quick test to verify RSI fix
"""

import pandas as pd
import numpy as np
import sys
import os

def test_rsi_fix():
    """Test that RSI calculation now works"""
    print("🔧 Testing RSI fix...")
    
    try:
        from smc_strategy import SMCStrategy
        import config
        
        # Create test data
        np.random.seed(42)
        data = []
        base_price = 0.08
        
        for i in range(60):
            price = base_price + np.random.normal(0, 0.002)
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price + np.random.normal(0, 0.001)
            close_price = price
            volume = np.random.uniform(800000, 1200000)
            
            data.append([open_price, high, low, close_price, volume])
        
        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
        
        strategy = SMCStrategy(config)
        
        # Test RSI calculation directly
        rsi_result = strategy._analyze_rsi(df)
        print(f"  ✅ RSI calculation works: {rsi_result}")
        
        # Test full signal generation
        signal_data = strategy.get_trading_signal(df)
        print(f"  ✅ Full signal generation works: {signal_data.get('signal', 'UNKNOWN')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RSI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing RSI Fix")
    print("=" * 30)
    
    if test_rsi_fix():
        print("✅ RSI fix successful!")
    else:
        print("❌ RSI fix failed!")
