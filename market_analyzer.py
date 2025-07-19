import requests
import pandas as pd
from datetime import datetime, timedelta
from smc_strategy import SMCStrategy
import config

class MarketAnalyzer:
    """
    Real-time market analyzer using SMC/ICT principles
    Fetches live data and provides high-confluence trade setups
    """
    
    def __init__(self):
        self.smc_strategy = SMCStrategy(config)
        self.base_url = "https://api.binance.com/api/v3"
        
    def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 200) -> pd.DataFrame:
        """Fetch candlestick data from Binance"""
        try:
            # Convert symbol format (BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace("/", "")
            
            url = f"{self.base_url}/klines"
            params = {
                "symbol": binance_symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            binance_symbol = symbol.replace("/", "")
            url = f"{self.base_url}/ticker/price"
            params = {"symbol": binance_symbol}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
            
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return 0.0
    
    def analyze_market_structure(self, symbol: str = "BTC/USDT", timeframes: list = None) -> dict:
        """
        Comprehensive market structure analysis across multiple timeframes
        """
        if timeframes is None:
            timeframes = ["4h", "1h", "15m"]
        
        analysis_results = {}
        
        for tf in timeframes:
            print(f"\n=== ANALYZING {symbol} on {tf.upper()} TIMEFRAME ===")
            
            # Fetch data
            df = self.fetch_klines(symbol, tf, 200)
            if df.empty:
                print(f"Failed to fetch data for {tf}")
                continue
            
            # Perform SMC analysis
            signal, analysis = self.smc_strategy.generate_smc_signal(df)
            
            # Calculate risk/reward levels
            rr_levels = self.smc_strategy.calculate_risk_reward_levels(signal, analysis)
            
            analysis_results[tf] = {
                "signal": signal,
                "analysis": analysis,
                "risk_reward": rr_levels,
                "current_price": analysis["current_price"]
            }
            
            # Print detailed analysis
            self._print_analysis(tf, signal, analysis, rr_levels)
        
        return analysis_results
    
    def _print_analysis(self, timeframe: str, signal: str, analysis: dict, rr_levels: dict):
        """Print formatted analysis results"""
        print(f"\n📊 {timeframe.upper()} TIMEFRAME ANALYSIS:")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"Signal: {signal}")
        
        # Market Structure
        ms = analysis["market_structure"]
        print(f"\n🏗️  MARKET STRUCTURE:")
        print(f"   Trend: {ms['trend']}")
        print(f"   Change of Character: {ms['choch']}")
        print(f"   Break of Structure: {ms['bos']}")
        
        # Order Blocks
        obs = analysis["order_blocks"]
        if obs:
            print(f"\n📦 ORDER BLOCKS (Top 3):")
            for i, ob in enumerate(obs[:3]):
                print(f"   {i+1}. {ob['type']} OB: ${ob['low']:.2f} - ${ob['high']:.2f} (Strength: {ob['strength']})")
        
        # Fair Value Gaps
        fvgs = analysis["fvgs"]
        if fvgs:
            print(f"\n⚡ FAIR VALUE GAPS:")
            for i, fvg in enumerate(fvgs[-3:]):
                print(f"   {i+1}. {fvg['type']} FVG: ${fvg['low']:.2f} - ${fvg['high']:.2f}")
        
        # Liquidity
        liq = analysis["liquidity"]
        print(f"\n💧 LIQUIDITY ZONES:")
        if liq["buy_side"]:
            print(f"   Buy-side (Resistance): ${liq['buy_side'][0]['level']:.2f}")
        if liq["sell_side"]:
            print(f"   Sell-side (Support): ${liq['sell_side'][0]['level']:.2f}")
        
        # Premium/Discount
        pd_zone = analysis["premium_discount"]
        print(f"\n🎯 PREMIUM/DISCOUNT: {pd_zone['zone']}")
        
        # Risk/Reward Setup
        if rr_levels and signal != "HOLD":
            print(f"\n💰 TRADE SETUP ({signal}):")
            print(f"   Entry: ${rr_levels['entry']:.2f}")
            print(f"   Stop Loss: ${rr_levels['stop_loss']:.2f}")
            print(f"   Take Profit: ${rr_levels['take_profit']:.2f}")
            print(f"   Risk/Reward: 1:{rr_levels['risk_reward']:.1f}")
            
            if rr_levels['risk_reward'] >= 5.0:
                print("   ✅ MEETS MINIMUM 1:5 RR CRITERIA")
            else:
                print("   ❌ DOES NOT MEET 1:5 RR CRITERIA")
    
    def find_high_probability_setup(self, symbol: str = "BTC/USDT") -> dict:
        """
        Find the highest probability trade setup across timeframes
        Only returns setups that meet strict confluence criteria
        """
        print(f"\n🔍 SEARCHING FOR HIGH-PROBABILITY SETUP ON {symbol}")
        print("=" * 60)
        
        # Analyze multiple timeframes
        analysis_results = self.analyze_market_structure(symbol)
        
        # Find confluence across timeframes
        signals = {}
        valid_setups = []
        
        for tf, result in analysis_results.items():
            signals[tf] = result["signal"]
            
            # Check if setup meets criteria
            if (result["signal"] != "HOLD" and 
                result["risk_reward"] and 
                result["risk_reward"].get("risk_reward", 0) >= 5.0):
                
                valid_setups.append({
                    "timeframe": tf,
                    "signal": result["signal"],
                    "setup": result["risk_reward"],
                    "confluence_score": self._calculate_confluence_score(result["analysis"])
                })
        
        print(f"\n🎯 CONFLUENCE ANALYSIS:")
        print(f"4H Signal: {signals.get('4h', 'N/A')}")
        print(f"1H Signal: {signals.get('1h', 'N/A')}")
        print(f"15M Signal: {signals.get('15m', 'N/A')}")
        
        # Check for multi-timeframe confluence
        buy_signals = sum(1 for s in signals.values() if s == "BUY")
        sell_signals = sum(1 for s in signals.values() if s == "SELL")
        
        print(f"\nBuy Signals: {buy_signals}/3")
        print(f"Sell Signals: {sell_signals}/3")
        
        # Select best setup
        if valid_setups:
            # Sort by confluence score
            best_setup = max(valid_setups, key=lambda x: x["confluence_score"])
            
            print(f"\n🚀 HIGH-PROBABILITY SETUP FOUND!")
            print(f"Timeframe: {best_setup['timeframe'].upper()}")
            print(f"Direction: {best_setup['signal']}")
            print(f"Confluence Score: {best_setup['confluence_score']}/10")
            
            setup = best_setup['setup']
            print(f"\n📋 TRADE PLAN:")
            print(f"Entry: ${setup['entry']:.2f}")
            print(f"Stop Loss: ${setup['stop_loss']:.2f}")
            print(f"Take Profit: ${setup['take_profit']:.2f}")
            print(f"Risk/Reward: 1:{setup['risk_reward']:.1f}")
            
            risk_amount = abs(setup['entry'] - setup['stop_loss'])
            reward_amount = abs(setup['take_profit'] - setup['entry'])
            print(f"Risk: ${risk_amount:.2f} | Reward: ${reward_amount:.2f}")
            
            return best_setup
        else:
            print(f"\n❌ NO HIGH-PROBABILITY SETUP FOUND")
            print("Waiting for better confluence and risk/reward opportunity...")
            return {}
    
    def _calculate_confluence_score(self, analysis: dict) -> float:
        """Calculate confluence score out of 10"""
        score = 0
        
        # Market structure (2 points)
        if analysis["market_structure"]["trend"] in ["BULLISH", "BEARISH"]:
            score += 2
        
        # Order blocks (2 points)
        strong_obs = [ob for ob in analysis["order_blocks"] if ob["strength"] >= 2]
        if strong_obs:
            score += 2
        
        # Fair value gaps (1 point)
        significant_fvgs = [fvg for fvg in analysis["fvgs"] if fvg["gap_size"] > 0.002]
        if significant_fvgs:
            score += 1
        
        # Premium/discount (1 point)
        if analysis["premium_discount"]["zone"] in ["PREMIUM", "DISCOUNT"]:
            score += 1
        
        # Liquidity (2 points)
        if (analysis["liquidity"]["buy_side"] and 
            analysis["liquidity"]["sell_side"]):
            score += 2
        
        # Change of character (2 points)
        if analysis["market_structure"]["choch"]:
            score += 2
        
        return min(score, 10)  # Cap at 10

# Example usage and live analysis
if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    
    print("🚀 SMART MONEY CONCEPTS (SMC) MARKET ANALYZER")
    print("=" * 50)
    
    # Analyze current market
    setup = analyzer.find_high_probability_setup("BTC/USDT")
    
    if setup:
        print(f"\n✅ TRADE RECOMMENDATION: {setup['signal']} {config.SYMBOL}")
        print("⚠️  Remember to:")
        print("   - Confirm setup on higher timeframes")
        print("   - Wait for proper entry confirmation")
        print("   - Manage risk according to your account size")
        print("   - Never risk more than 1-2% per trade")
    else:
        print(f"\n⏳ NO TRADE RECOMMENDED AT THIS TIME")
        print("   Continue monitoring for high-confluence setups")
