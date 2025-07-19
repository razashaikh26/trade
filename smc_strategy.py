import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import pandas_ta as ta

class SMCStrategy:
    """
    Advanced Smart Money Concepts (SMC) and ICT Trading Strategy
    Incorporates Order Blocks, Fair Value Gaps, Liquidity Analysis, and Market Structure
    """
    
    def __init__(self, config):
        self.config = config
        self.lookback_period = 50
        self.order_block_strength = 3  # Minimum touches for valid order block
        self.fvg_threshold = 0.001  # Minimum gap size as percentage
        self.liquidity_threshold = 0.002  # Minimum liquidity zone size
        
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
    
    def generate_smc_signal(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Generate trading signal based on SMC analysis"""
        if len(df) < 50:
            return "HOLD", {"reason": "Insufficient data"}
        
        # Perform all SMC analysis
        market_structure = self.identify_market_structure(df)
        order_blocks = self.identify_order_blocks(df)
        fvgs = self.identify_fair_value_gaps(df)
        liquidity = self.identify_liquidity_zones(df)
        premium_discount = self.calculate_premium_discount(df)
        
        current_price = df['close'].iloc[-1]
        
        analysis = {
            "market_structure": market_structure,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
            "liquidity": liquidity,
            "premium_discount": premium_discount,
            "current_price": current_price
        }
        
        # Signal generation logic
        signal = self._evaluate_confluence(analysis)
        
        return signal, analysis
    
    def _evaluate_confluence(self, analysis: Dict) -> str:
        """Evaluate confluence of SMC factors for signal generation"""
        bullish_factors = 0
        bearish_factors = 0
        
        # Market structure
        if analysis["market_structure"]["trend"] == "BULLISH":
            bullish_factors += 2
        elif analysis["market_structure"]["trend"] == "BEARISH":
            bearish_factors += 2
            
        # Change of Character
        if analysis["market_structure"]["choch"]:
            if analysis["market_structure"]["trend"] == "BULLISH":
                bearish_factors += 3  # CHoCH suggests reversal
            else:
                bullish_factors += 3
        
        # Order Blocks
        current_price = analysis["current_price"]
        for ob in analysis["order_blocks"]:
            if ob["strength"] >= 2:  # Strong order block
                if (ob["type"] == "BULLISH" and 
                    ob["low"] <= current_price <= ob["high"]):
                    bullish_factors += 3
                elif (ob["type"] == "BEARISH" and 
                      ob["low"] <= current_price <= ob["high"]):
                    bearish_factors += 3
        
        # Fair Value Gaps
        for fvg in analysis["fvgs"]:
            if fvg["gap_size"] > 0.002:  # Significant gap
                if (fvg["type"] == "BULLISH" and 
                    fvg["low"] <= current_price <= fvg["high"]):
                    bullish_factors += 2
                elif (fvg["type"] == "BEARISH" and 
                      fvg["low"] <= current_price <= fvg["high"]):
                    bearish_factors += 2
        
        # Premium/Discount
        if analysis["premium_discount"]["zone"] == "DISCOUNT":
            bullish_factors += 1  # Buy in discount
        elif analysis["premium_discount"]["zone"] == "PREMIUM":
            bearish_factors += 1  # Sell in premium
        
        # Liquidity considerations
        # Look for liquidity sweeps (price taking out stops then reversing)
        buy_side_liq = analysis["liquidity"]["buy_side"]
        sell_side_liq = analysis["liquidity"]["sell_side"]
        
        if buy_side_liq and current_price > buy_side_liq[0]["level"]:
            bearish_factors += 2  # Swept buy stops, expect reversal down
        if sell_side_liq and current_price < sell_side_liq[0]["level"]:
            bullish_factors += 2  # Swept sell stops, expect reversal up
        
        # Decision logic (require high confluence)
        if bullish_factors >= 5 and bullish_factors > bearish_factors + 2:
            return "BUY"
        elif bearish_factors >= 5 and bearish_factors > bullish_factors + 2:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_risk_reward_levels(self, signal: str, analysis: Dict) -> Dict:
        """Calculate entry, SL, and TP levels with minimum 1:5 RR"""
        if signal == "HOLD":
            return {}
            
        current_price = analysis["current_price"]
        
        if signal == "BUY":
            # Entry at current price or better
            entry = current_price
            
            # Stop loss BELOW entry (correct for LONG)
            sl_candidates = []
            
            # Check order blocks
            for ob in analysis["order_blocks"]:
                if ob["type"] == "BULLISH" and ob["low"] < current_price:
                    sl_candidates.append(ob["low"] * 0.999)  # Slightly below
            
            # Check swing lows
            if analysis["market_structure"]["swing_lows"]:
                recent_low = min(analysis["market_structure"]["swing_lows"][-2:])
                if recent_low < current_price:
                    sl_candidates.append(recent_low * 0.999)
            
            if not sl_candidates:
                # Use percentage-based SL as fallback (BELOW entry for LONG)
                sl = current_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
            else:
                sl = max(sl_candidates)  # Tightest stop loss (highest value below entry)
            
            # Take profit ABOVE entry (correct for LONG)
            risk = entry - sl
            min_reward = risk * 5  # 1:5 risk/reward ratio
            tp = entry + min_reward  # ADD reward to entry for LONG
            
            # Check if TP hits resistance
            resistance_levels = []
            for ob in analysis["order_blocks"]:
                if ob["type"] == "BEARISH" and ob["high"] > current_price:
                    resistance_levels.append(ob["high"])
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                if tp > nearest_resistance:
                    tp = nearest_resistance * 0.999  # Just below resistance
            
            # SANITY CHECK for LONG position
            if tp <= entry or sl >= entry:
                print(f"ERROR: Invalid TP/SL for LONG position! Entry: {entry}, TP: {tp}, SL: {sl}")
                # Auto-correct using percentage-based approach
                sl = entry * (1 - self.config.STOP_LOSS_PERCENT / 100)
                tp = entry * (1 + self.config.TAKE_PROFIT_PERCENT / 100)
                print(f"Auto-corrected: Entry: {entry}, TP: {tp}, SL: {sl}")
            
            rr_ratio = (tp - entry) / (entry - sl) if entry != sl else 0
            
            return {
                "entry": round(entry, 2),
                "stop_loss": round(sl, 2),
                "take_profit": round(tp, 2),
                "rr_ratio": round(rr_ratio, 2)
            }
            
        elif signal == "SELL":
            # Entry at current price or better
            entry = current_price
            
            # Stop loss ABOVE entry (correct for SHORT)
            sl_candidates = []
            
            # Check order blocks
            for ob in analysis["order_blocks"]:
                if ob["type"] == "BEARISH" and ob["high"] > current_price:
                    sl_candidates.append(ob["high"] * 1.001)  # Slightly above
            
            # Check swing highs
            if analysis["market_structure"]["swing_highs"]:
                recent_high = max(analysis["market_structure"]["swing_highs"][-2:])
                if recent_high > current_price:
                    sl_candidates.append(recent_high * 1.001)
            
            if not sl_candidates:
                # Use percentage-based SL as fallback (ABOVE entry for SHORT)
                sl = current_price * (1 + self.config.STOP_LOSS_PERCENT / 100)
            else:
                sl = min(sl_candidates)  # Tightest stop loss (lowest value above entry)
            
            # Take profit BELOW entry (correct for SHORT)
            risk = sl - entry
            min_reward = risk * 5  # 1:5 risk/reward ratio
            tp = entry - min_reward  # SUBTRACT reward from entry for SHORT
            
            # Check if TP hits support
            support_levels = []
            for ob in analysis["order_blocks"]:
                if ob["type"] == "BULLISH" and ob["low"] < current_price:
                    support_levels.append(ob["low"])
            
            if support_levels:
                nearest_support = max(support_levels)
                if tp < nearest_support:
                    tp = nearest_support * 1.001  # Just above support
            
            # SANITY CHECK for SHORT position
            if tp >= entry or sl <= entry:
                print(f"ERROR: Invalid TP/SL for SHORT position! Entry: {entry}, TP: {tp}, SL: {sl}")
                # Auto-correct using percentage-based approach
                sl = entry * (1 + self.config.STOP_LOSS_PERCENT / 100)
                tp = entry * (1 - self.config.TAKE_PROFIT_PERCENT / 100)
                print(f"Auto-corrected: Entry: {entry}, TP: {tp}, SL: {sl}")
            
            rr_ratio = (entry - tp) / (sl - entry) if sl != entry else 0
            
            return {
                "entry": round(entry, 2),
                "stop_loss": round(sl, 2),
                "take_profit": round(tp, 2),
                "rr_ratio": round(rr_ratio, 2)
            }
        
        return {}
