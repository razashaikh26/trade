import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import pandas_ta as ta
from technical_indicators import TechnicalIndicators

class SMCStrategy:
    """
    Advanced Smart Money Concepts (SMC) and ICT Trading Strategy
    Incorporates Order Blocks, Fair Value Gaps, Liquidity Analysis, and Market Structure
    Enhanced with ATR-based dynamic levels and candlestick pattern confirmation
    """
    
    def __init__(self, config, testnet=False):
        self.config = config
        self.testnet = testnet
        self.lookback_period = 50
        self.order_block_strength = 3  # Minimum touches for valid order block
        self.fvg_threshold = 0.001  # Minimum gap size as percentage
        self.liquidity_threshold = 0.002  # Minimum liquidity zone size
        self.tech_indicators = TechnicalIndicators()
        
    def _is_market_tradeable(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if the market has enough volume and volatility to trade"""
        # Volatility Check (ATR)
        atr = self.tech_indicators.calculate_atr(df, period=self.config.ATR_PERIOD)
        if atr.empty or pd.isna(atr.iloc[-1]):
            return False, "Unable to calculate ATR"
            
        current_atr = atr.iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_percent = (current_atr / current_price) * 100
        
        if atr_percent < self.config.MIN_ATR_PERCENT:
            return False, f"Low volatility (ATR: {atr_percent:.2f}%)"
        
        # Volume Check
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_ma
        
        if volume_ratio < self.config.MIN_VOLUME_RATIO:
            return False, f"Low volume (Ratio: {volume_ratio:.2f})"
            
        return True, "Market is tradeable"

    def identify_market_structure(self, df: pd.DataFrame) -> Dict:
        """Identify market structure: HH, HL, LH, LL"""
        if len(df) < 20:
            return {"trend": "UNKNOWN", "bos": False, "choch": False}
            
        # Calculate swing highs and lows
        highs = df['high'].rolling(window=5, center=True).max() == df['high']
        lows = df['low'].rolling(window=5, center=True).min() == df['low']
        
        swing_highs = df[highs]['high'].tail(3)
        swing_lows = df[lows]['low'].tail(3)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"trend": "UNKNOWN", "bos": False, "choch": False}
        
        # Determine trend
        recent_highs = swing_highs.tail(2)
        recent_lows = swing_lows.tail(2)
        
        higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-2]
        higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-2]
        lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-2]
        lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-2]
        
        if higher_highs and higher_lows:
            trend = "BULLISH"
        elif lower_highs and lower_lows:
            trend = "BEARISH"
        else:
            trend = "RANGING"
            
        # Check for Break of Structure (BOS) or Change of Character (CHoCH)
        current_price = df['close'].iloc[-1]
        
        bos = False
        choch = False
        
        if trend == "BULLISH" and len(swing_lows) >= 2:
            last_low = swing_lows.iloc[-2]
            if current_price < last_low:
                choch = True
        elif trend == "BEARISH" and len(swing_highs) >= 2:
            last_high = swing_highs.iloc[-2]
            if current_price > last_high:
                choch = True
                
        return {
            "trend": trend,
            "bos": bos,
            "choch": choch,
            "swing_highs": swing_highs.tolist(),
            "swing_lows": swing_lows.tolist()
        }
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify institutional order blocks"""
        order_blocks = []
        
        if len(df) < 20:
            return order_blocks
            
        # Look for strong moves (>1% in single candle)
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        strong_moves = df[df['body_size'] > 0.01]
        
        for i, idx in enumerate(strong_moves.index[-10:]):  # Last 10 strong moves
            idx_pos = df.index.get_loc(idx)
            if idx_pos < 5 or idx_pos >= len(df) - 1:
                continue
                
            candle = df.loc[idx]
            
            # Bullish order block (strong green candle)
            if candle['close'] > candle['open']:
                ob_high = candle['high']
                ob_low = candle['low']
                ob_type = "BULLISH"
                
                # Check if price has returned to this zone
                future_data = df.iloc[idx_pos+1:]
                if len(future_data) > 0 and future_data['low'].min() <= ob_high:
                    order_blocks.append({
                        "type": ob_type,
                        "high": ob_high,
                        "low": ob_low,
                        "index": idx_pos,
                        "strength": self._calculate_ob_strength(df, idx_pos, ob_high, ob_low)
                    })
            
            # Bearish order block (strong red candle)
            elif candle['close'] < candle['open']:
                ob_high = candle['high']
                ob_low = candle['low']
                ob_type = "BEARISH"
                
                # Check if price has returned to this zone
                future_data = df.iloc[idx_pos+1:]
                if len(future_data) > 0 and future_data['high'].max() >= ob_low:
                    order_blocks.append({
                        "type": ob_type,
                        "high": ob_high,
                        "low": ob_low,
                        "index": idx_pos,
                        "strength": self._calculate_ob_strength(df, idx_pos, ob_high, ob_low)
                    })
        
        # Sort by strength and return top 5
        return sorted(order_blocks, key=lambda x: x['strength'], reverse=True)[:5]
    
    def _calculate_ob_strength(self, df: pd.DataFrame, idx: int, high: float, low: float) -> float:
        """Calculate order block strength based on reactions"""
        if idx >= len(df) - 5:
            return 0
            
        future_data = df.iloc[idx+1:idx+10]
        reactions = 0
        
        for _, candle in future_data.iterrows():
            if low <= candle['low'] <= high or low <= candle['high'] <= high:
                reactions += 1
                
        return reactions
    
    def identify_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Identify Fair Value Gaps (imbalances)"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
            
        for i in range(1, len(df) - 1):
            prev_candle = df.iloc[i-1]
            curr_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Bullish FVG (gap up)
            if (prev_candle['high'] < next_candle['low'] and 
                curr_candle['close'] > curr_candle['open']):
                gap_size = (next_candle['low'] - prev_candle['high']) / prev_candle['high']
                if gap_size > self.fvg_threshold:
                    fvgs.append({
                        "type": "BULLISH",
                        "high": next_candle['low'],
                        "low": prev_candle['high'],
                        "index": i,
                        "gap_size": gap_size
                    })
            
            # Bearish FVG (gap down)
            elif (prev_candle['low'] > next_candle['high'] and 
                  curr_candle['close'] < curr_candle['open']):
                gap_size = (prev_candle['low'] - next_candle['high']) / next_candle['high']
                if gap_size > self.fvg_threshold:
                    fvgs.append({
                        "type": "BEARISH",
                        "high": prev_candle['low'],
                        "low": next_candle['high'],
                        "index": i,
                        "gap_size": gap_size
                    })
        
        return fvgs[-5:]  # Return last 5 FVGs
    
    def identify_liquidity_zones(self, df: pd.DataFrame) -> Dict:
        """Identify buy-side and sell-side liquidity"""
        if len(df) < 20:
            return {"buy_side": [], "sell_side": []}
            
        # Find recent highs and lows that likely have stops
        highs = df['high'].rolling(window=10, center=True).max() == df['high']
        lows = df['low'].rolling(window=10, center=True).min() == df['low']
        
        recent_highs = df[highs]['high'].tail(5)
        recent_lows = df[lows]['low'].tail(5)
        
        buy_side_liquidity = []  # Above highs (buy stops)
        sell_side_liquidity = []  # Below lows (sell stops)
        
        for high in recent_highs:
            buy_side_liquidity.append({
                "level": high,
                "type": "BUY_STOPS",
                "strength": self._calculate_liquidity_strength(df, high, "high")
            })
            
        for low in recent_lows:
            sell_side_liquidity.append({
                "level": low,
                "type": "SELL_STOPS", 
                "strength": self._calculate_liquidity_strength(df, low, "low")
            })
        
        return {
            "buy_side": sorted(buy_side_liquidity, key=lambda x: x['strength'], reverse=True),
            "sell_side": sorted(sell_side_liquidity, key=lambda x: x['strength'], reverse=True)
        }
    
    def _calculate_liquidity_strength(self, df: pd.DataFrame, level: float, level_type: str) -> float:
        """Calculate liquidity strength based on touches and volume"""
        touches = 0
        
        if level_type == "high":
            touches = len(df[df['high'] >= level * 0.999])  # Within 0.1%
        else:
            touches = len(df[df['low'] <= level * 1.001])   # Within 0.1%
            
        return touches
    
    def calculate_premium_discount(self, df: pd.DataFrame) -> Dict:
        """Calculate premium/discount zones using Fibonacci"""
        if len(df) < 50:
            return {"zone": "UNKNOWN", "fib_levels": {}}
            
        # Get recent swing high and low
        recent_data = df.tail(50)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        current_price = df['close'].iloc[-1]
        
        # Calculate Fibonacci levels
        diff = swing_high - swing_low
        fib_levels = {
            "0.0": swing_low,
            "0.236": swing_low + (diff * 0.236),
            "0.382": swing_low + (diff * 0.382),
            "0.5": swing_low + (diff * 0.5),
            "0.618": swing_low + (diff * 0.618),
            "0.786": swing_low + (diff * 0.786),
            "1.0": swing_high
        }
        
        # Determine if current price is in premium or discount
        if current_price > fib_levels["0.618"]:
            zone = "PREMIUM"
        elif current_price < fib_levels["0.382"]:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"
            
        return {"zone": zone, "fib_levels": fib_levels}
    
    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        """
        Enhanced main method to get trading signals with ATR-based levels and pattern confirmation
        """
        if len(df) < 50:
            return {"signal": "HOLD", "reason": "Insufficient data", "confidence": 0}
        
        # Check if market is tradeable
        tradeable, reason = self._is_market_tradeable(df)
        if not tradeable:
            return {"signal": "HOLD", "reason": reason, "confidence": 0}
        
        # Calculate ATR and volatility metrics
        atr = self.tech_indicators.calculate_atr(df, period=self.config.ATR_PERIOD)
        atr_ma = atr.rolling(window=self.config.ATR_MA_PERIOD).mean()
        
        current_atr = atr.iloc[-1] if not atr.empty else 0
        current_atr_ma = atr_ma.iloc[-1] if not atr_ma.empty else current_atr
        current_price = df['close'].iloc[-1]
        
        # Get volatility state
        volatility_state = self.tech_indicators.get_market_volatility_state(current_atr, current_atr_ma)
        
        # Check for engulfing patterns if enabled
        engulfing_patterns = {}
        if getattr(self.config, 'USE_ENGULFING_FILTER', False):
            engulfing_patterns = self.tech_indicators.detect_engulfing_patterns(df)
        
        # Get base SMC signal
        base_signal = self._get_base_smc_signal(df)
        
        if base_signal["signal"] == "HOLD":
            return base_signal
        
        # Apply engulfing pattern filter
        if getattr(self.config, 'USE_ENGULFING_FILTER', False):
            signal_direction = base_signal["signal"]
            
            # For BUY signals, check for bullish engulfing confirmation
            if signal_direction == "BUY" and not engulfing_patterns.get('bullish_engulfing', False):
                return {
                    "signal": "HOLD", 
                    "reason": "BUY signal but no bullish engulfing confirmation",
                    "confidence": 0
                }
            
            # For SELL signals, check for bearish engulfing confirmation
            if signal_direction == "SELL" and not engulfing_patterns.get('bearish_engulfing', False):
                return {
                    "signal": "HOLD", 
                    "reason": "SELL signal but no bearish engulfing confirmation",
                    "confidence": 0
                }
        
        # Calculate dynamic ATR-based levels
        dynamic_levels = {}
        if getattr(self.config, 'USE_DYNAMIC_ATR_LEVELS', False) and current_atr > 0:
            atr_multiplier = self.tech_indicators.calculate_dynamic_atr_multiplier(volatility_state)
            dynamic_levels = self.tech_indicators.atr_based_levels(
                current_price, current_atr, atr_multiplier, testnet=self.testnet
            )
        
        # Calculate adaptive position size
        adaptive_position_size = None
        if getattr(self.config, 'USE_ADAPTIVE_POSITION_SIZE', False):
            # This would need account balance - will be calculated in main.py
            adaptive_position_size = {
                'current_atr': current_atr,
                'atr_ma': current_atr_ma,
                'volatility_state': volatility_state
            }
        
        # Enhanced signal with new features
        enhanced_signal = {
            **base_signal,
            'atr_data': {
                'current_atr': current_atr,
                'atr_ma': current_atr_ma,
                'volatility_state': volatility_state,
                'atr_percent': (current_atr / current_price) * 100
            },
            'engulfing_patterns': engulfing_patterns,
            'dynamic_levels': dynamic_levels,
            'adaptive_sizing': adaptive_position_size
        }
        
        return enhanced_signal
    
    def _get_base_smc_signal(self, df: pd.DataFrame) -> Dict:
        """
        Get base SMC trading signal (original logic)
        """
        # Market structure analysis
        market_structure = self.identify_market_structure(df)
        
        # Order block analysis
        order_blocks = self.identify_order_blocks(df)
        
        # Fair Value Gap analysis
        fvgs = self.identify_fair_value_gaps(df)
        
        # Liquidity analysis
        liquidity_zones = self.identify_liquidity_zones(df)
        
        # RSI analysis
        rsi_signal = self._analyze_rsi(df)
        
        # Moving average trend
        ma_trend = self._analyze_moving_average(df)
        
        # Combine all signals
        signal_strength = 0
        reasons = []
        
        # Market structure signals
        if market_structure["bos"] or market_structure["choch"]:
            if market_structure["trend"] == "BULLISH":
                signal_strength += 2
                reasons.append("Bullish market structure break")
            elif market_structure["trend"] == "BEARISH":
                signal_strength -= 2
                reasons.append("Bearish market structure break")
        
        # Order block signals
        current_price = df['close'].iloc[-1]
        for ob in order_blocks:
            if ob["type"] == "bullish" and current_price <= ob["high"] and current_price >= ob["low"]:
                signal_strength += 1
                reasons.append("Price at bullish order block")
            elif ob["type"] == "bearish" and current_price <= ob["high"] and current_price >= ob["low"]:
                signal_strength -= 1
                reasons.append("Price at bearish order block")
        
        # Fair Value Gap signals
        for fvg in fvgs:
            if fvg["type"] == "bullish" and current_price >= fvg["low"] and current_price <= fvg["high"]:
                signal_strength += 1
                reasons.append("Price in bullish FVG")
            elif fvg["type"] == "bearish" and current_price >= fvg["low"] and current_price <= fvg["high"]:
                signal_strength -= 1
                reasons.append("Price in bearish FVG")
        
        # RSI signals
        if rsi_signal["signal"] == "BUY":
            signal_strength += 1
            reasons.append(f"RSI oversold: {rsi_signal['value']:.1f}")
        elif rsi_signal["signal"] == "SELL":
            signal_strength -= 1
            reasons.append(f"RSI overbought: {rsi_signal['value']:.1f}")
        
        # Moving average signals
        if ma_trend["signal"] == "BUY":
            signal_strength += 1
            reasons.append("Price above MA trend")
        elif ma_trend["signal"] == "SELL":
            signal_strength -= 1
            reasons.append("Price below MA trend")
        
        # Determine final signal
        if signal_strength >= 3:
            signal = "BUY"
            confidence = min(signal_strength * 20, 100)
        elif signal_strength <= -3:
            signal = "SELL" 
            confidence = min(abs(signal_strength) * 20, 100)
        else:
            signal = "HOLD"
            confidence = 0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reason": "; ".join(reasons) if reasons else "No clear signal",
            "market_structure": market_structure,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
            "liquidity_zones": liquidity_zones,
            "rsi": rsi_signal,
            "ma_trend": ma_trend,
            "signal_strength": signal_strength
        }

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI for overbought/oversold conditions"""
        import pandas_ta as ta
        
        # Use proper pandas-ta syntax
        rsi = ta.rsi(df['close'], length=14)
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30:
            return {"signal": "BUY", "value": current_rsi}
        elif current_rsi > 70:
            return {"signal": "SELL", "value": current_rsi}
        else:
            return {"signal": "NEUTRAL", "value": current_rsi}

    def _analyze_moving_average(self, df: pd.DataFrame) -> Dict:
        """Analyze moving average trend"""
        short_ma = df['close'].rolling(window=20).mean()
        long_ma = df['close'].rolling(window=50).mean()
        
        current_price = df['close'].iloc[-1]
        short_ma_value = short_ma.iloc[-1]
        long_ma_value = long_ma.iloc[-1]
        
        if current_price > short_ma_value and short_ma_value > long_ma_value:
            return {"signal": "BUY", "short_ma": short_ma_value, "long_ma": long_ma_value}
        elif current_price < short_ma_value and short_ma_value < long_ma_value:
            return {"signal": "SELL", "short_ma": short_ma_value, "long_ma": long_ma_value}
        else:
            return {"signal": "NEUTRAL", "short_ma": short_ma_value, "long_ma": long_ma_value}
