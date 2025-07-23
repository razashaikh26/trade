import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import pandas_ta as ta
from technical_indicators import TechnicalIndicators
import logging

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
        """Check if market conditions are suitable for trading"""
        try:
            # Check if we have enough data
            if len(df) < 20:  # Minimum required candles
                logging.warning("⚠️ Not enough data points for analysis")
                return False, "Insufficient data"

            # Get current price and ATR
            current_price = df['close'].iloc[-1]
            atr = self.tech_indicators.calculate_atr(df, period=14)
            current_atr = atr.iloc[-1]
            
            # Calculate ATR as percentage of price
            atr_percent = (current_atr / current_price) * 100
            
            logging.info(f"📊 Market Tradeability Check:")
            logging.info(f"   Current Price: {current_price:.6f}")
            logging.info(f"   ATR: {current_atr:.6f} ({atr_percent:.2f}% of price)")
            logging.info(f"   Minimum ATR% required: {self.config.MIN_ATR_PERCENT}%")
            
            # Check ATR volatility
            if atr_percent < self.config.MIN_ATR_PERCENT:
                logging.warning(f"   ⚠️ Low volatility (ATR: {atr_percent:.2f}% < {self.config.MIN_ATR_PERCENT}%)")
                return False, f"Low volatility (ATR: {atr_percent:.2f}%)"
            
            # Check volume
            volume_ma = df['volume'].rolling(window=20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 0
            
            logging.info(f"   Current Volume: {current_volume:.2f}")
            logging.info(f"   20-period Avg Volume: {volume_ma.iloc[-1]:.2f}")
            logging.info(f"   Volume Ratio: {volume_ratio:.4f} (Min: {self.config.MIN_VOLUME_RATIO})")
            
            if volume_ratio < self.config.MIN_VOLUME_RATIO:
                logging.warning(f"   ⚠️ Volume below threshold: {volume_ratio:.4f} < {self.config.MIN_VOLUME_RATIO}")
                return False, f"Volume below threshold ({volume_ratio:.2f} < {self.config.MIN_VOLUME_RATIO})"
            
            # Check if market is open (not during weekends or holidays)
            # This is a simplified check - you might want to use a proper market calendar
            current_time = pd.Timestamp.now()
            if current_time.weekday() >= 5:  # Weekend
                logging.warning(f"   ⚠️ Market closed (weekend)")
                return False, "Market closed (weekend)"
            
            # Check if within trading hours (24/7 for crypto)
            logging.info("   ✅ Market is tradeable")
            return True, "Market is tradeable"
            
        except Exception as e:
            logging.error(f"❌ Error checking market tradeability: {str(e)}", exc_info=True)
            return False, f"Error: {str(e)}"
    
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
        Generate trading signals with enhanced entry conditions
        """
        try:
            # Add technical indicators
            df = df.copy()
            
            # Calculate RSI using pandas_ta's rsi function
            df['rsi'] = ta.rsi(df['close'], length=self.config.RSI_LENGTH)
            
            # Calculate SMA using pandas_ta's sma function
            df['sma_20'] = ta.sma(df['close'], length=20)
            
            # Add volume confirmation
            df = TechnicalIndicators.add_volume_confirmation(df)
            
            # Detect candlestick patterns
            df = TechnicalIndicators.detect_candlestick_patterns(df)
            
            # Get market regime
            market_regime = TechnicalIndicators.get_market_regime(df)
            
            # Initialize signal
            signal = "HOLD"
            confidence = 0
            reasons = []
            
            # Get latest data point
            latest = df.iloc[-1]
            
            # Volume confirmation
            volume_confirmed = latest.get('volume_confirmed', False)
            
            # Check for buy signals
            buy_conditions = [
                latest['rsi'] < self.config.RSI_OVERSOLD if pd.notna(latest['rsi']) else False,
                latest['close'] > latest['sma_20'] if pd.notna(latest['sma_20']) else False,
                latest.get('bullish_engulfing', False) or latest.get('hammer', False),
                volume_confirmed
            ]
            
            # Check for sell signals
            sell_conditions = [
                latest['rsi'] > self.config.RSI_OVERBOUGHT if pd.notna(latest['rsi']) else False,
                latest['close'] < latest['sma_20'] if pd.notna(latest['sma_20']) else False,
                latest.get('bearish_engulfing', False) or latest.get('shooting_star', False),
                volume_confirmed
            ]
            
            # Adjust confidence based on number of conditions met
            buy_score = sum([1 for cond in buy_conditions if cond is True])
            sell_score = sum([1 for cond in sell_conditions if cond is True])
            
            # Generate signal based on conditions and market regime
            if buy_score >= 3:  # At least 3 out of 4 conditions
                signal = "BUY"
                confidence = min(100, 50 + (buy_score * 10))  # 80-90% confidence
                reasons.append(f"Bullish setup: {buy_score}/4 conditions met")
                
                # Adjust for market regime
                if market_regime == 'trending':
                    confidence += 5
                    reasons.append("Trending market favors trend-following")
                
            elif sell_score >= 3:
                signal = "SELL"
                confidence = min(100, 50 + (sell_score * 10))
                reasons.append(f"Bearish setup: {sell_score}/4 conditions met")
                
                # Adjust for market regime
                if market_regime == 'trending':
                    confidence += 5
                    reasons.append("Trending market favors trend-following")
            
            # Add market regime to reasons
            reasons.append(f"Market regime: {market_regime.upper()}")
            
            # Add volume status
            reasons.append(f"Volume {'confirmed' if volume_confirmed else 'below average'}")
            
            # Add RSI value if available
            if pd.notna(latest['rsi']):
                reasons.append(f"RSI: {latest['rsi']:.1f}")
            
            # Log the final decision
            logging.info(f"Signal: {signal} (Confidence: {confidence}%)")
            for reason in reasons:
                logging.info(f" - {reason}")
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasons': reasons,
                'market_regime': market_regime,
                'volume_confirmed': volume_confirmed,
                'rsi': latest.get('rsi'),
                'price': latest['close']
            }
            
        except Exception as e:
            logging.error(f"Error generating trading signal: {e}", exc_info=True)
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reasons': [f"Error: {str(e)}"]
            }

    def _get_base_smc_signal(self, df: pd.DataFrame) -> Dict:
        """Generate base trading signal using Smart Money Concepts"""
        # Market structure analysis
        market_structure = self.identify_market_structure(df)
        logging.info(f"🔍 Market Structure: {market_structure}")
        
        # Order block analysis
        order_blocks = self.identify_order_blocks(df)
        logging.info(f"🔍 Order Blocks: {len(order_blocks)} found")
        
        # Fair Value Gap analysis
        fvgs = self.identify_fair_value_gaps(df)
        logging.info(f"🔍 Fair Value Gaps: {len(fvgs)} found")
        
        # RSI analysis
        rsi_signal = self._analyze_rsi(df)
        
        # Moving average trend
        ma_trend = self._analyze_moving_average(df)
        
        # Combine all signals
        signal_strength = 0
        reasons = []
        
        # 1. Market structure signals (strongest weight)
        if market_structure["bos"] or market_structure["choch"]:
            if market_structure["trend"] == "BULLISH":
                signal_strength += 3  # Increased from 2
                reasons.append("Bullish market structure break")
                logging.info("✅ Added +3 for Bullish market structure break")
            elif market_structure["trend"] == "BEARISH":
                signal_strength -= 3  # Increased from 2
                reasons.append("Bearish market structure break")
                logging.info("✅ Added -3 for Bearish market structure break")
        
        # 2. RSI signals (strong weight)
        if rsi_signal["signal"] == "BUY":
            # Add more weight if RSI is significantly oversold
            rsi_value = rsi_signal["value"]
            rsi_strength = max(1, (self.config.RSI_OVERSOLD - rsi_value) / 2)
            signal_strength += rsi_strength
            reasons.append(f"RSI oversold: {rsi_value:.1f} (strength: +{rsi_strength:.1f})")
            logging.info(f"✅ Added +{rsi_strength:.1f} for RSI oversold: {rsi_value:.1f}")
        elif rsi_signal["signal"] == "SELL":
            rsi_value = rsi_signal["value"]
            rsi_strength = max(1, (rsi_value - self.config.RSI_OVERBOUGHT) / 2)
            signal_strength -= rsi_strength
            reasons.append(f"RSI overbought: {rsi_value:.1f} (strength: -{rsi_strength:.1f})")
            logging.info(f"✅ Added -{rsi_strength:.1f} for RSI overbought: {rsi_value:.1f}")
        
        # 3. Moving average signals (medium weight)
        if ma_trend["signal"] == "BUY":
            signal_strength += 1.5  # Increased from 1
            reasons.append("Price above MA trend")
            logging.info(f"✅ Added +1.5 for Price above MA ({ma_trend['ma_value']:.6f})")
        elif ma_trend["signal"] == "SELL":
            signal_strength -= 1.5  # Increased from 1
            reasons.append("Price below MA trend")
            logging.info(f"✅ Added -1.5 for Price below MA ({ma_trend['ma_value']:.6f})")
        
        # 4. Order block signals (weaker weight)
        current_price = df['close'].iloc[-1]
        for ob in order_blocks:
            if ob["type"] == "bullish" and current_price <= ob["high"] and current_price >= ob["low"]:
                signal_strength += 0.5  # Reduced from 1
                reasons.append("Price at bullish order block")
                logging.info(f"✅ Added +0.5 for Bullish order block at {ob['low']}-{ob['high']}")
            elif ob["type"] == "bearish" and current_price <= ob["high"] and current_price >= ob["low"]:
                signal_strength -= 0.5  # Reduced from 1
                reasons.append("Price at bearish order block")
                logging.info(f"✅ Added -0.5 for Bearish order block at {ob['low']}-{ob['high']}")
        
        # 5. FVG signals (weaker weight)
        for fvg in fvgs:
            if fvg["type"] == "bullish" and current_price >= fvg["low"] and current_price <= fvg["high"]:
                signal_strength += 0.5  # Reduced from 1
                reasons.append("Price in bullish FVG")
                logging.info(f"✅ Added +0.5 for Bullish FVG at {fvg['low']}-{fvg['high']}")
            elif fvg["type"] == "bearish" and current_price >= fvg["low"] and current_price <= fvg["high"]:
                signal_strength -= 0.5  # Reduced from 1
                reasons.append("Price in bearish FVG")
                logging.info(f"✅ Added -0.5 for Bearish FVG at {fvg['low']}-{fvg['high']}")
        
        logging.info(f"🔢 Final Signal Strength: {signal_strength:.2f}")
        
        # Determine final signal with dynamic confidence
        if signal_strength > 0.5:  # Lowered threshold from 1
            signal = "BUY"
            confidence = min(abs(signal_strength) * 40, 100)  # Increased multiplier from 30
            logging.info(f"🎯 FINAL SIGNAL: BUY (Confidence: {confidence:.1f}%)")
        elif signal_strength < -0.5:  # Lowered threshold from -1
            signal = "SELL"
            confidence = min(abs(signal_strength) * 40, 100)  # Increased multiplier from 30
            logging.info(f"🎯 FINAL SIGNAL: SELL (Confidence: {confidence:.1f}%)")
        else:
            signal = "HOLD"
            confidence = 0
            logging.info("🎯 FINAL SIGNAL: HOLD (No clear signal)")
        
        return {
            "signal": signal,
            "confidence": confidence,
            "reason": "; ".join(reasons) if reasons else "No clear signal",
            "market_structure": market_structure,
            "order_blocks": order_blocks,
            "fvgs": fvgs,
            "rsi": rsi_signal,
            "ma_trend": ma_trend,
            "signal_strength": signal_strength
        }

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI for overbought/oversold conditions"""
        import pandas_ta as ta
        
        # Use configured RSI settings instead of hardcoded values
        rsi = ta.rsi(df['close'], length=self.config.RSI_LENGTH)
        current_rsi = rsi.iloc[-1]
        
        # Debug logging
        logging.info(f" RSI Analysis: Current RSI={current_rsi:.1f}, Oversold={self.config.RSI_OVERSOLD}, Overbought={self.config.RSI_OVERBOUGHT}")
        
        if current_rsi < self.config.RSI_OVERSOLD:
            logging.info(f" RSI BUY Signal: {current_rsi:.1f} < {self.config.RSI_OVERSOLD}")
            return {"signal": "BUY", "value": current_rsi}
        elif current_rsi > self.config.RSI_OVERBOUGHT:
            logging.info(f" RSI SELL Signal: {current_rsi:.1f} > {self.config.RSI_OVERBOUGHT}")
            return {"signal": "SELL", "value": current_rsi}
        else:
            logging.info(f" RSI NEUTRAL: {current_rsi:.1f} between {self.config.RSI_OVERSOLD}-{self.config.RSI_OVERBOUGHT}")
            return {"signal": "NEUTRAL", "value": current_rsi}

    def _analyze_moving_average(self, df: pd.DataFrame) -> Dict:
        """Analyze moving average trend with detailed logging"""
        # Use configured MA period
        ma = df['close'].rolling(window=self.config.MA_PERIOD).mean()
        current_price = df['close'].iloc[-1]
        ma_value = ma.iloc[-1]
        
        logging.info(f"📈 MA Analysis (Period={self.config.MA_PERIOD}):")
        logging.info(f"   Current Price: {current_price:.6f}")
        logging.info(f"   MA Value: {ma_value:.6f}")
        logging.info(f"   Price vs MA: {current_price - ma_value:.6f} ({'ABOVE' if current_price > ma_value else 'BELOW' if current_price < ma_value else 'AT'} MA)")
        
        # Determine trend
        if current_price > ma_value:
            logging.info(f"   ✅ Price is ABOVE MA - Bullish Signal")
            return {"signal": "BUY", "ma_value": ma_value}
        elif current_price < ma_value:
            logging.info(f"   ✅ Price is BELOW MA - Bearish Signal")
            return {"signal": "SELL", "ma_value": ma_value}
        else:
            logging.info(f"   ⏸️  Price is AT MA - Neutral")
            return {"signal": "NEUTRAL", "ma_value": ma_value}
