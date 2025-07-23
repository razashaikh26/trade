import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import simple_indicators as talib  # Use our simple indicators instead of TA-Lib
from technical_indicators import TechnicalIndicators

class SMCStrategy:
    def __init__(self, config):
        self.config = config
        self.tech_indicators = TechnicalIndicators()
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.last_trade_time = 0
        self.open_positions = {}
        
    def analyze_market(self, df, symbol):
        """Enhanced market analysis with multiple confirmations"""
        try:
            if len(df) < 50:
                return 'HOLD', 0, 'Insufficient data for analysis'
            
            # Check daily loss limit
            if self.daily_pnl <= -self.config.DAILY_LOSS_LIMIT:
                return 'HOLD', 0, f'Daily loss limit reached: {self.daily_pnl:.2f}%'
            
            # Check time between trades
            import time
            current_time = time.time()
            if current_time - self.last_trade_time < self.config.MIN_TIME_BETWEEN_TRADES:
                return 'HOLD', 0, 'Minimum time between trades not met'
            
            # Check maximum concurrent positions
            if len(self.open_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
                return 'HOLD', 0, 'Maximum concurrent positions reached'
            
            # Calculate technical indicators
            signals = self._calculate_enhanced_signals(df)
            
            # Market regime detection
            regime = self._detect_market_regime(df) if self.config.USE_MARKET_REGIME else 'unknown'
            
            # Trend confirmation
            trend_confirmed = self._confirm_trend(df) if self.config.USE_TREND_CONFIRMATION else True
            
            # Volume confirmation
            volume_confirmed = self._confirm_volume(df)
            
            # Volatility filter
            volatility_ok = self._check_volatility(df)
            
            # RSI divergence check
            rsi_signal = self._check_rsi_divergence(df)
            
            # Candlestick pattern confirmation
            pattern_confirmed = True
            if self.config.USE_ENGULFING_FILTER:
                pattern_confirmed = self._check_engulfing_pattern(df)
            
            # Combine all signals for final decision
            final_signal, confidence, analysis = self._combine_signals(
                signals, regime, trend_confirmed, volume_confirmed, 
                volatility_ok, rsi_signal, pattern_confirmed
            )
            
            return final_signal, confidence, analysis
            
        except Exception as e:
            return 'HOLD', 0, f'Error in market analysis: {str(e)}'
    
    def _calculate_enhanced_signals(self, df):
        """Calculate multiple technical indicator signals"""
        signals = {}
        
        # SMC Structure Analysis (Advanced)
        signals['smc'] = self._analyze_smc_structure(df)
        
        # BEGINNER-FRIENDLY STRATEGIES
        if self.config.USE_RSI_BUY_LOW_SELL_HIGH:
            signals['rsi_buy_low_sell_high'] = self._analyze_rsi_buy_low_sell_high(df)
        
        if self.config.USE_SMA_EMA_CROSSOVER:
            signals['sma_ema_crossover'] = self._analyze_sma_ema_crossover(df)
        
        if self.config.USE_MACD_CROSSOVER:
            signals['macd_crossover'] = self._analyze_macd_crossover(df)
        
        if self.config.USE_SUPPORT_RESISTANCE:
            signals['support_resistance'] = self._analyze_support_resistance(df)
        
        if self.config.USE_VOLUME_SPIKE_FILTER:
            signals['volume_spike'] = self._analyze_volume_spike(df)
        
        # ADVANCED STRATEGIES (from previous implementation)
        if self.config.USE_TREND_CONFIRMATION:
            signals['ma'] = self._analyze_moving_averages(df)
        
        signals['rsi'] = self._analyze_rsi(df)
        signals['macd'] = self._analyze_macd(df)
        signals['bb'] = self._analyze_bollinger_bands(df)
        signals['stoch'] = self._analyze_stochastic(df)
        
        return signals
    
    # BEGINNER-FRIENDLY STRATEGY IMPLEMENTATIONS
    
    def _analyze_rsi_buy_low_sell_high(self, df):
        """RSI-Based Buy Low, Sell High Strategy (Beginner Friendly & Reliable)"""
        try:
            rsi = talib.RSI(df['close'].values, timeperiod=self.config.RSI_PERIOD)
            current_rsi = rsi[-1]
            
            # Look at RSI trend over last few periods
            rsi_trend = rsi[-self.config.RSI_LOOKBACK_PERIODS:]
            
            score = 0
            
            # Strong buy signals (oversold conditions)
            if current_rsi <= self.config.RSI_EXTREME_OVERSOLD:
                score += 3  # Very strong buy signal
            elif current_rsi <= self.config.RSI_BUY_LOW_THRESHOLD:
                score += 2  # Strong buy signal
            elif current_rsi <= 40:
                score += 1  # Mild buy signal
            
            # Strong sell signals (overbought conditions)
            elif current_rsi >= self.config.RSI_EXTREME_OVERBOUGHT:
                score -= 3  # Very strong sell signal
            elif current_rsi >= self.config.RSI_SELL_HIGH_THRESHOLD:
                score -= 2  # Strong sell signal
            elif current_rsi >= 60:
                score -= 1  # Mild sell signal
            
            # RSI trend confirmation
            if len(rsi_trend) >= 2:
                if current_rsi < self.config.RSI_BUY_LOW_THRESHOLD and rsi_trend[-1] > rsi_trend[-2]:
                    score += 0.5  # RSI turning up from oversold
                elif current_rsi > self.config.RSI_SELL_HIGH_THRESHOLD and rsi_trend[-1] < rsi_trend[-2]:
                    score -= 0.5  # RSI turning down from overbought
            
            return score
            
        except Exception:
            return 0
    
    def _analyze_sma_ema_crossover(self, df):
        """SMA/EMA Crossover Strategy (Trend Following)"""
        try:
            if self.config.USE_EMA_OVER_SMA:
                # Use EMA crossover
                fast_ma = talib.EMA(df['close'].values, timeperiod=self.config.EMA_FAST_PERIOD)
                slow_ma = talib.EMA(df['close'].values, timeperiod=self.config.EMA_SLOW_PERIOD)
            else:
                # Use SMA crossover
                fast_ma = talib.SMA(df['close'].values, timeperiod=self.config.SMA_FAST_PERIOD)
                slow_ma = talib.SMA(df['close'].values, timeperiod=self.config.SMA_SLOW_PERIOD)
            
            score = 0
            
            # Current crossover status
            if fast_ma[-1] > slow_ma[-1]:
                score += 1  # Bullish crossover
            else:
                score -= 1  # Bearish crossover
            
            # Check for recent crossover (more reliable signal)
            confirmation_periods = self.config.CROSSOVER_CONFIRMATION_PERIODS
            if len(fast_ma) > confirmation_periods:
                # Bullish crossover confirmation
                if (fast_ma[-1] > slow_ma[-1] and 
                    fast_ma[-confirmation_periods] <= slow_ma[-confirmation_periods]):
                    score += 2  # Strong bullish signal
                
                # Bearish crossover confirmation
                elif (fast_ma[-1] < slow_ma[-1] and 
                      fast_ma[-confirmation_periods] >= slow_ma[-confirmation_periods]):
                    score -= 2  # Strong bearish signal
            
            # MA slope confirmation
            if len(fast_ma) >= 2:
                if fast_ma[-1] > fast_ma[-2]:  # Fast MA trending up
                    score += 0.5
                else:  # Fast MA trending down
                    score -= 0.5
            
            return score
            
        except Exception:
            return 0
    
    def _analyze_macd_crossover(self, df):
        """MACD Signal Line Crossover Strategy"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=self.config.MACD_FAST,
                slowperiod=self.config.MACD_SLOW,
                signalperiod=self.config.MACD_SIGNAL
            )
            
            score = 0
            
            if len(macd) > 1:
                # MACD line vs Signal line crossover
                if macd[-1] > macd_signal[-1]:
                    score += self.config.MACD_SIGNAL_CROSSOVER_WEIGHT
                else:
                    score -= self.config.MACD_SIGNAL_CROSSOVER_WEIGHT
                
                # Recent crossover detection (stronger signal)
                if (macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]):
                    score += 1.5  # Bullish crossover
                elif (macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]):
                    score -= 1.5  # Bearish crossover
                
                # MACD zero line cross
                if macd[-1] > 0:
                    score += self.config.MACD_ZERO_LINE_WEIGHT
                else:
                    score -= self.config.MACD_ZERO_LINE_WEIGHT
                
                # MACD histogram momentum
                if macd_hist[-1] > macd_hist[-2]:
                    score += self.config.MACD_HISTOGRAM_WEIGHT
                else:
                    score -= self.config.MACD_HISTOGRAM_WEIGHT
            
            return score
            
        except Exception:
            return 0
    
    def _analyze_support_resistance(self, df):
        """Support & Resistance Zones Strategy (S/R Bounce)"""
        try:
            if len(df) < self.config.SR_LOOKBACK_PERIODS:
                return 0
            
            current_price = df['close'].iloc[-1]
            
            # Find support and resistance levels
            support_levels = self._find_support_levels(df)
            resistance_levels = self._find_resistance_levels(df)
            
            score = 0
            
            # Check for support bounce (buy signal)
            for support in support_levels:
                price_diff = abs(current_price - support['level']) / support['level']
                if price_diff <= self.config.SR_TOUCH_TOLERANCE:
                    # Price is near support level
                    bounce_strength = support['strength'] * self.config.SR_STRENGTH_MULTIPLIER
                    
                    # Check for bounce confirmation
                    if self._confirm_support_bounce(df, support['level']):
                        score += bounce_strength
                        break
            
            # Check for resistance rejection (sell signal)
            for resistance in resistance_levels:
                price_diff = abs(current_price - resistance['level']) / resistance['level']
                if price_diff <= self.config.SR_TOUCH_TOLERANCE:
                    # Price is near resistance level
                    rejection_strength = resistance['strength'] * self.config.SR_STRENGTH_MULTIPLIER
                    
                    # Check for rejection confirmation
                    if self._confirm_resistance_rejection(df, resistance['level']):
                        score -= rejection_strength
                        break
            
            return score
            
        except Exception:
            return 0
    
    def _analyze_volume_spike(self, df):
        """Volume Spike Confirmation (Anti-Fakeout Filter)"""
        try:
            if 'volume' not in df.columns or len(df) < self.config.VOLUME_SPIKE_LOOKBACK:
                return 0
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=self.config.VOLUME_SPIKE_LOOKBACK).mean().iloc[-1]
            
            # Check minimum volume requirement
            if current_volume < self.config.MIN_VOLUME_FOR_TRADE:
                return -1  # Penalize low volume
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            score = 0
            
            # Volume spike confirmation
            if volume_ratio >= self.config.VOLUME_SPIKE_MULTIPLIER:
                score += 2  # Strong volume confirmation
            elif volume_ratio >= 1.2:
                score += 1  # Moderate volume confirmation
            elif volume_ratio < 0.8:
                score -= 1  # Below average volume (penalty)
            
            # Volume trend analysis
            if len(df) >= 3:
                recent_volumes = df['volume'].tail(3).values
                if recent_volumes[-1] > recent_volumes[-2] > recent_volumes[-3]:
                    score += 0.5  # Increasing volume trend
                elif recent_volumes[-1] < recent_volumes[-2] < recent_volumes[-3]:
                    score -= 0.5  # Decreasing volume trend
            
            return score
            
        except Exception:
            return 0
    
    # HELPER METHODS FOR SUPPORT/RESISTANCE
    
    def _find_support_levels(self, df):
        """Find support levels using swing lows"""
        support_levels = []
        lookback = self.config.SR_LOOKBACK_PERIODS
        
        try:
            # Find swing lows
            for i in range(5, len(df) - 5):
                if i >= lookback:
                    break
                    
                current_low = df['low'].iloc[i]
                is_swing_low = True
                
                # Check if it's a swing low
                for j in range(i-5, i+6):
                    if j != i and df['low'].iloc[j] < current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    # Count touches to this level
                    touches = self._count_level_touches(df, current_low, 'support')
                    if touches >= self.config.SR_MIN_TOUCHES:
                        support_levels.append({
                            'level': current_low,
                            'strength': touches,
                            'index': i
                        })
            
            # Sort by strength
            return sorted(support_levels, key=lambda x: x['strength'], reverse=True)[:3]
            
        except Exception:
            return []
    
    def _find_resistance_levels(self, df):
        """Find resistance levels using swing highs"""
        resistance_levels = []
        lookback = self.config.SR_LOOKBACK_PERIODS
        
        try:
            # Find swing highs
            for i in range(5, len(df) - 5):
                if i >= lookback:
                    break
                    
                current_high = df['high'].iloc[i]
                is_swing_high = True
                
                # Check if it's a swing high
                for j in range(i-5, i+6):
                    if j != i and df['high'].iloc[j] > current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    # Count touches to this level
                    touches = self._count_level_touches(df, current_high, 'resistance')
                    if touches >= self.config.SR_MIN_TOUCHES:
                        resistance_levels.append({
                            'level': current_high,
                            'strength': touches,
                            'index': i
                        })
            
            # Sort by strength
            return sorted(resistance_levels, key=lambda x: x['strength'], reverse=True)[:3]
            
        except Exception:
            return []
    
    def _count_level_touches(self, df, level, level_type):
        """Count how many times price touched a support/resistance level"""
        touches = 0
        tolerance = self.config.SR_TOUCH_TOLERANCE
        
        try:
            for i in range(len(df)):
                if level_type == 'support':
                    price = df['low'].iloc[i]
                    if abs(price - level) / level <= tolerance:
                        touches += 1
                else:  # resistance
                    price = df['high'].iloc[i]
                    if abs(price - level) / level <= tolerance:
                        touches += 1
            
            return touches
            
        except Exception:
            return 0
    
    def _confirm_support_bounce(self, df, support_level):
        """Confirm if price is bouncing from support"""
        try:
            confirmation_periods = self.config.SR_BOUNCE_CONFIRMATION
            if len(df) < confirmation_periods:
                return False
            
            recent_candles = df.tail(confirmation_periods)
            
            # Check if price touched support and is now moving up
            touched_support = False
            moving_up = False
            
            for _, candle in recent_candles.iterrows():
                if abs(candle['low'] - support_level) / support_level <= self.config.SR_TOUCH_TOLERANCE:
                    touched_support = True
                    break
            
            if touched_support:
                # Check if recent candles are moving up
                if recent_candles['close'].iloc[-1] > recent_candles['close'].iloc[-2]:
                    moving_up = True
            
            return touched_support and moving_up
            
        except Exception:
            return False
    
    def _confirm_resistance_rejection(self, df, resistance_level):
        """Confirm if price is being rejected from resistance"""
        try:
            confirmation_periods = self.config.SR_BOUNCE_CONFIRMATION
            if len(df) < confirmation_periods:
                return False
            
            recent_candles = df.tail(confirmation_periods)
            
            # Check if price touched resistance and is now moving down
            touched_resistance = False
            moving_down = False
            
            for _, candle in recent_candles.iterrows():
                if abs(candle['high'] - resistance_level) / resistance_level <= self.config.SR_TOUCH_TOLERANCE:
                    touched_resistance = True
                    break
            
            if touched_resistance:
                # Check if recent candles are moving down
                if recent_candles['close'].iloc[-1] < recent_candles['close'].iloc[-2]:
                    moving_down = True
            
            return touched_resistance and moving_down
            
        except Exception:
            return False
    
    def _analyze_smc_structure(self, df):
        """Enhanced SMC (Smart Money Concepts) analysis"""
        try:
            # Calculate swing highs and lows
            swing_highs = self._find_swing_points(df['high'], 'high')
            swing_lows = self._find_swing_points(df['low'], 'low')
            
            # Identify market structure shifts
            structure_shift = self._identify_structure_shift(swing_highs, swing_lows)
            
            # Find order blocks
            order_blocks = self._find_order_blocks(df)
            
            # Identify fair value gaps
            fvg = self._find_fair_value_gaps(df)
            
            # Liquidity analysis
            liquidity_sweep = self._check_liquidity_sweep(df, swing_highs, swing_lows)
            
            # Combine SMC signals
            smc_score = 0
            if structure_shift == 'bullish':
                smc_score += 2
            elif structure_shift == 'bearish':
                smc_score -= 2
                
            if order_blocks == 'bullish':
                smc_score += 1
            elif order_blocks == 'bearish':
                smc_score -= 1
                
            if fvg == 'bullish':
                smc_score += 1
            elif fvg == 'bearish':
                smc_score -= 1
                
            if liquidity_sweep == 'bullish':
                smc_score += 1
            elif liquidity_sweep == 'bearish':
                smc_score -= 1
            
            return smc_score
            
        except Exception:
            return 0
    
    def _analyze_moving_averages(self, df):
        """Enhanced moving average analysis"""
        try:
            # Calculate multiple MAs
            ma_fast = df['close'].rolling(window=self.config.TREND_MA_FAST).mean()
            ma_slow = df['close'].rolling(window=self.config.TREND_MA_SLOW).mean()
            ma_filter = df['close'].rolling(window=self.config.TREND_MA_FILTER).mean()
            
            current_price = df['close'].iloc[-1]
            
            score = 0
            # Fast MA vs Slow MA
            if ma_fast.iloc[-1] > ma_slow.iloc[-1]:
                score += 1
            else:
                score -= 1
                
            # Price vs Filter MA
            if current_price > ma_filter.iloc[-1]:
                score += 1
            else:
                score -= 1
                
            # MA slope analysis
            if ma_fast.iloc[-1] > ma_fast.iloc[-2]:
                score += 0.5
            else:
                score -= 0.5
                
            return score
            
        except Exception:
            return 0
    
    def _analyze_rsi(self, df):
        """Enhanced RSI analysis with divergence"""
        try:
            rsi = talib.RSI(df['close'].values, timeperiod=self.config.RSI_PERIOD)
            current_rsi = rsi[-1]
            
            score = 0
            # Basic RSI signals
            if current_rsi < self.config.RSI_OVERSOLD:
                score += 2  # Strong buy signal
            elif current_rsi < 40:
                score += 1  # Mild buy signal
            elif current_rsi > self.config.RSI_OVERBOUGHT:
                score -= 2  # Strong sell signal
            elif current_rsi > 60:
                score -= 1  # Mild sell signal
                
            # RSI momentum
            if len(rsi) > 1 and rsi[-1] > rsi[-2]:
                score += 0.5
            elif len(rsi) > 1 and rsi[-1] < rsi[-2]:
                score -= 0.5
                
            return score
            
        except Exception:
            return 0
    
    def _analyze_macd(self, df):
        """MACD analysis"""
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=self.config.MACD_FAST,
                slowperiod=self.config.MACD_SLOW,
                signalperiod=self.config.MACD_SIGNAL
            )
            
            score = 0
            if len(macd) > 1:
                # MACD line vs Signal line
                if macd[-1] > macd_signal[-1]:
                    score += 1
                else:
                    score -= 1
                    
                # MACD histogram momentum
                if macd_hist[-1] > macd_hist[-2]:
                    score += 0.5
                else:
                    score -= 0.5
                    
                # Zero line cross
                if macd[-1] > 0:
                    score += 0.5
                else:
                    score -= 0.5
                    
            return score
            
        except Exception:
            return 0
    
    def _analyze_bollinger_bands(self, df):
        """Bollinger Bands analysis"""
        try:
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=self.config.BB_PERIOD,
                nbdevup=self.config.BB_STD_DEV,
                nbdevdn=self.config.BB_STD_DEV
            )
            
            current_price = df['close'].iloc[-1]
            
            score = 0
            # Price position within bands
            if current_price < lower[-1]:
                score += 2  # Oversold
            elif current_price < middle[-1]:
                score += 1  # Below middle
            elif current_price > upper[-1]:
                score -= 2  # Overbought
            elif current_price > middle[-1]:
                score -= 1  # Above middle
                
            # Band squeeze detection
            band_width = (upper[-1] - lower[-1]) / middle[-1]
            avg_band_width = np.mean([(upper[i] - lower[i]) / middle[i] for i in range(-20, -1)])
            
            if band_width < avg_band_width * 0.8:
                score += 0.5  # Squeeze suggests breakout coming
                
            return score
            
        except Exception:
            return 0
    
    def _analyze_stochastic(self, df):
        """Stochastic oscillator analysis"""
        try:
            slowk, slowd = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=self.config.STOCH_K,
                slowk_period=self.config.STOCH_D,
                slowd_period=self.config.STOCH_D
            )
            
            score = 0
            if len(slowk) > 1:
                current_k = slowk[-1]
                current_d = slowd[-1]
                
                # Oversold/Overbought levels
                if current_k < self.config.STOCH_OVERSOLD:
                    score += 2
                elif current_k > self.config.STOCH_OVERBOUGHT:
                    score -= 2
                    
                # %K vs %D crossover
                if current_k > current_d and slowk[-2] <= slowd[-2]:
                    score += 1  # Bullish crossover
                elif current_k < current_d and slowk[-2] >= slowd[-2]:
                    score -= 1  # Bearish crossover
                    
            return score
            
        except Exception:
            return 0
    
    def _detect_market_regime(self, df):
        """Detect market regime (trending/ranging)"""
        try:
            # Calculate ADX for trend strength
            adx = talib.ADX(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=14
            )
            
            current_adx = adx[-1]
            
            if current_adx > self.config.TRENDING_THRESHOLD * 100:
                return 'trending'
            elif current_adx < self.config.RANGING_THRESHOLD * 100:
                return 'ranging'
            else:
                return 'neutral'
                
        except Exception:
            return 'unknown'
    
    def _confirm_trend(self, df):
        """Multi-timeframe trend confirmation"""
        try:
            # Use multiple MA periods for trend confirmation
            ma_short = df['close'].rolling(window=self.config.TREND_MA_FAST).mean()
            ma_long = df['close'].rolling(window=self.config.TREND_MA_SLOW).mean()
            ma_filter = df['close'].rolling(window=self.config.TREND_MA_FILTER).mean()
            
            current_price = df['close'].iloc[-1]
            
            # All MAs should be aligned for strong trend
            current_price = df['close'].iloc[-1]
            
            # Bullish trend: price > ma_short > ma_long > ma_filter
            bullish_trend = (current_price > ma_short.iloc[-1] > 
                           ma_long.iloc[-1] > ma_filter.iloc[-1])
            
            # Bearish trend: price < ma_short < ma_long < ma_filter
            bearish_trend = (current_price < ma_short.iloc[-1] < 
                           ma_long.iloc[-1] < ma_filter.iloc[-1])
            
            return bullish_trend or bearish_trend
            
        except Exception:
            return True  # Default to confirmed if calculation fails
    
    def _confirm_volume(self, df):
        """Volume confirmation"""
        try:
            if 'volume' not in df.columns:
                return True  # Skip if no volume data
                
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=self.config.VOLUME_MA_PERIOD).mean().iloc[-1]
            
            # Volume should be above average for confirmation
            return current_volume > avg_volume * (1 + self.config.MIN_VOLUME_RATIO)
            
        except Exception:
            return True
    
    def _check_volatility(self, df):
        """Check if volatility is sufficient for trading"""
        try:
            atr = self.tech_indicators.calculate_atr(df, period=self.config.ATR_PERIOD)
            current_atr = atr.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            atr_percent = (current_atr / current_price) * 100
            
            return atr_percent >= self.config.MIN_ATR_PERCENT
            
        except Exception:
            return True
    
    def _check_rsi_divergence(self, df):
        """Check for RSI divergence"""
        try:
            rsi = talib.RSI(df['close'].values, timeperiod=self.config.RSI_PERIOD)
            
            # Look for divergence in last N periods
            lookback = min(self.config.RSI_DIVERGENCE_LOOKBACK, len(df) - 1)
            
            price_trend = df['close'].iloc[-1] - df['close'].iloc[-lookback]
            rsi_trend = rsi[-1] - rsi[-lookback]
            
            # Bullish divergence: price down, RSI up
            if price_trend < 0 and rsi_trend > 0:
                return 1
            # Bearish divergence: price up, RSI down
            elif price_trend > 0 and rsi_trend < 0:
                return -1
            else:
                return 0
                
        except Exception:
            return 0
    
    def _check_engulfing_pattern(self, df):
        """Check for engulfing candlestick patterns"""
        try:
            if len(df) < 2:
                return True
                
            # Get last two candles
            prev_candle = df.iloc[-2]
            current_candle = df.iloc[-1]
            
            # Bullish engulfing
            bullish_engulfing = (
                prev_candle['close'] < prev_candle['open'] and  # Previous red
                current_candle['close'] > current_candle['open'] and  # Current green
                current_candle['open'] < prev_candle['close'] and  # Gap down open
                current_candle['close'] > prev_candle['open']  # Engulfs previous
            )
            
            # Bearish engulfing
            bearish_engulfing = (
                prev_candle['close'] > prev_candle['open'] and  # Previous green
                current_candle['close'] < current_candle['open'] and  # Current red
                current_candle['open'] > prev_candle['close'] and  # Gap up open
                current_candle['close'] < prev_candle['open']  # Engulfs previous
            )
            
            return bullish_engulfing or bearish_engulfing
            
        except Exception:
            return True
    
    def _combine_signals(self, signals, regime, trend_confirmed, volume_confirmed, 
                        volatility_ok, rsi_signal, pattern_confirmed):
        """Combine all signals for final decision"""
        try:
            # Calculate total score
            total_score = 0
            max_score = 0
            
            for signal_name, score in signals.items():
                if signal_name == 'smc':
                    weight = 3  # SMC gets highest weight
                elif signal_name in ['ma', 'rsi']:
                    weight = 2  # Important indicators
                else:
                    weight = 1  # Supporting indicators
                    
                total_score += score * weight
                max_score += 4 * weight  # Assuming max score per indicator is 4
            
            # Add RSI divergence
            total_score += rsi_signal * 2
            max_score += 4
            
            # Apply filters
            if not trend_confirmed:
                total_score *= 0.5
                
            if not volume_confirmed:
                total_score *= 0.7
                
            if not volatility_ok:
                return 'HOLD', 0, 'Insufficient volatility'
                
            if not pattern_confirmed:
                total_score *= 0.8
            
            # Adjust for market regime
            if regime == 'trending':
                total_score *= 1.2  # Boost signals in trending market
            elif regime == 'ranging':
                total_score *= 0.8  # Reduce signals in ranging market
            
            # Adjust for consecutive losses
            if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
                total_score *= self.config.DRAWDOWN_REDUCTION_FACTOR
            
            # Calculate confidence
            confidence = min(abs(total_score) / max_score * 100, 100)
            
            # Determine signal
            if total_score > 2:
                signal = 'BUY'
            elif total_score < -2:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Create analysis summary
            analysis = f"Score: {total_score:.1f}, Regime: {regime}, Trend: {trend_confirmed}, Vol: {volume_confirmed}"
            
            return signal, confidence, analysis
            
        except Exception as e:
            return 'HOLD', 0, f'Error combining signals: {str(e)}'
    
    # Helper methods for SMC analysis
    def _find_swing_points(self, series, point_type, window=5):
        """Find swing highs and lows"""
        swings = []
        for i in range(window, len(series) - window):
            if point_type == 'high':
                if all(series[i] >= series[j] for j in range(i-window, i+window+1) if j != i):
                    swings.append((i, series[i]))
            else:  # low
                if all(series[i] <= series[j] for j in range(i-window, i+window+1) if j != i):
                    swings.append((i, series[i]))
        return swings
    
    def _identify_structure_shift(self, swing_highs, swing_lows):
        """Identify market structure shifts"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral'
            
        # Check for higher highs and higher lows (bullish)
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]
        
        if (recent_highs[1][1] > recent_highs[0][1] and 
            recent_lows[1][1] > recent_lows[0][1]):
            return 'bullish'
        elif (recent_highs[1][1] < recent_highs[0][1] and 
              recent_lows[1][1] < recent_lows[0][1]):
            return 'bearish'
        else:
            return 'neutral'
    
    def _find_order_blocks(self, df):
        """Find order blocks (simplified)"""
        try:
            # Look for strong moves followed by consolidation
            if len(df) < 10:
                return 'neutral'
                
            recent_data = df.tail(10)
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if price_change > 0.02:  # 2% move up
                return 'bullish'
            elif price_change < -0.02:  # 2% move down
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _find_fair_value_gaps(self, df):
        """Find fair value gaps (simplified)"""
        try:
            if len(df) < 3:
                return 'neutral'
                
            # Check last 3 candles for gaps
            for i in range(len(df) - 3, len(df) - 1):
                current = df.iloc[i]
                next_candle = df.iloc[i + 1]
                
                # Bullish FVG: gap up
                if next_candle['low'] > current['high']:
                    return 'bullish'
                # Bearish FVG: gap down
                elif next_candle['high'] < current['low']:
                    return 'bearish'
                    
            return 'neutral'
            
        except Exception:
            return 'neutral'
    
    def _check_liquidity_sweep(self, df, swing_highs, swing_lows):
        """Check for liquidity sweeps"""
        try:
            if len(df) < 5 or not swing_highs or not swing_lows:
                return 'neutral'
                
            current_price = df['close'].iloc[-1]
            recent_high = max([h[1] for h in swing_highs[-3:]] if len(swing_highs) >= 3 else [swing_highs[-1][1]])
            recent_low = min([l[1] for l in swing_lows[-3:]] if len(swing_lows) >= 3 else [swing_lows[-1][1]])
            
            # Check if price swept above recent high then reversed
            if current_price > recent_high * 1.001:  # 0.1% above
                return 'bearish'  # Potential liquidity grab
            elif current_price < recent_low * 0.999:  # 0.1% below
                return 'bullish'  # Potential liquidity grab
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
