import os
import time
import config
import pandas as pd
import ccxt
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

class BinanceClient:
    def __init__(self, mock_mode=False, testnet=False):
        # Load environment variables
        load_dotenv()
        
        # Initialize Binance client using CCXT
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API key and secret must be set in .env file")
            
        self.client = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # important to avoid getting banned
            'options': {
                'defaultType': 'future', # Use 'future' for DOGE futures trading
            },
        })

        # Set testnet mode if enabled
        if testnet:
            self.client.set_sandbox_mode(True)
        
        # Store latest prices
        self.latest_prices = {}
        self.websocket_running = False
        
        # Mock mode settings
        self.mock_mode = mock_mode
        self.mock_balance = {'USDT': 1000.0}  # Default mock balance of 1000 USDT
        
    def get_account_balance(self, asset='USDT'):
        """Get account balance for a specific asset"""
        if self.mock_mode:
            # Return mock balance if in mock mode
            return self.mock_balance.get(asset, 0.0)
        
        try:
            # Get futures account balance
            balance = self.client.fetch_balance()
            
            # For futures, check both 'free' and 'total' balance
            if asset in balance:
                # Return available balance for trading
                available = balance[asset]['free'] if balance[asset]['free'] > 0 else balance[asset]['total']
                return float(available)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error getting balance: {e}")
            # If there's an error, return mock balance to continue testing
            return 100.0  # Return small test balance
    
    def get_klines(self, symbol, interval, limit=100):
        """Get historical klines (candlestick data)"""
        if self.mock_mode:
            # Generate mock klines data for testing
            import numpy as np
            
            # Current timestamp in milliseconds
            now_ms = int(time.time() * 1000)
            
            # Convert interval to minutes for timestamp calculation
            interval_minutes = 0
            if interval.endswith('m'):
                interval_minutes = int(interval[:-1])
            elif interval.endswith('h'):
                interval_minutes = int(interval[:-1]) * 60
            elif interval.endswith('d'):
                interval_minutes = int(interval[:-1]) * 60 * 24
            else:
                interval_minutes = 15  # Default to 15m
            
            # Generate timestamps for each candle
            timestamps = [now_ms - (i * interval_minutes * 60 * 1000) for i in range(limit-1, -1, -1)]
            
            # Generate mock price data (random walk starting at current BTC price)
            base_price = 30000  # Default BTC price if we can't get real price
            try:
                real_price = self.get_latest_price(symbol)
                if real_price:
                    base_price = real_price
            except:
                pass
            
            # Generate random price movements
            np.random.seed(42)  # For reproducible results
            price_changes = np.random.normal(0, base_price * 0.01, limit).cumsum()
            
            # Create OHLCV data
            ohlcv = []
            for i in range(limit):
                timestamp = timestamps[i]
                close = max(100, base_price + price_changes[i])  # Ensure price doesn't go too low
                open_price = close * (1 + np.random.normal(0, 0.005))
                high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.003)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.003)))
                volume = abs(np.random.normal(10, 5))
                
                ohlcv.append([timestamp, open_price, high, low, close, volume])
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        try:
            # Convert interval to CCXT format (e.g., '15m' is already correct)
            timeframe = interval
            
            # Fetch OHLCV data
            ohlcv = self.client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
            return df
        except Exception as e:
            print(f"Error getting klines: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol):
        """Get latest price for a symbol"""
        if self.mock_mode:
            # For mock mode, return a realistic price for the symbol
            if symbol == 'BTCUSDT':
                return 30000.0 + (time.time() % 1000)  # Slight variation based on time
            elif symbol == 'ETHUSDT':
                return 2000.0 + (time.time() % 100)
            else:
                return 100.0  # Default price for other symbols
        
        try:
            ticker = self.client.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"Error getting latest price: {e}")
            return None
    
    def place_order(self, symbol, side, quantity, order_type='market', price=None, stop_loss=None, take_profit=None):
        """Place an order on Binance"""
        # Handle mock mode
        if self.mock_mode:
            if not price:
                price = self.get_latest_price(symbol)
                if not price:
                    return None
            
            mock_order = {
                'id': 'mock-order-' + str(int(time.time())),
                'symbol': symbol,
                'type': order_type,
                'side': side.lower(),
                'amount': quantity,
                'price': price,
                'status': 'closed',
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.now().isoformat(),
                'info': {'mock': True}
            }
            
            if side.lower() == 'buy':
                cost = quantity * price
                self.mock_balance['USDT'] = self.mock_balance.get('USDT', 0) - cost
                symbol_asset = symbol.replace('USDT', '')
                self.mock_balance[symbol_asset] = self.mock_balance.get(symbol_asset, 0) + quantity
            else:  # sell
                proceeds = quantity * price
                self.mock_balance['USDT'] = self.mock_balance.get('USDT', 0) + proceeds
                symbol_asset = symbol.replace('USDT', '')
                self.mock_balance[symbol_asset] = self.mock_balance.get(symbol_asset, 0) - quantity
            
            return mock_order

        # Real mode
        try:
            # Create the primary order. We ignore stop_loss and take_profit here
            # because they must be created in separate orders after this one fills.
            order = self.client.create_order(
                symbol=symbol,
                type=order_type, # This is 'market' as set in main.py
                side=side,
                amount=quantity,
                price=price, # ccxt handles price=None for market orders
                params={}    # Pass empty params to prevent ccxt from being too smart
            )

            # TODO: After the market order fills, create separate stop-loss and take-profit orders.

            return order
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def get_open_positions(self, symbol):
        """Get open positions for a symbol. Returns a list of positions."""
        if self.mock_mode:
            return []
        try:
            all_positions = self.client.fetch_positions()
            # Binance symbols can be tricky (e.g., 'BTC/USDT' vs 'BTCUSDT'). We'll check for both.
            normalized_symbol = symbol.replace('/', '')
            for position in all_positions:
                pos_symbol = position.get('info', {}).get('symbol')
                contracts = float(position.get('contracts', 0))
                if pos_symbol == normalized_symbol and contracts != 0:
                    return [position] # Return a list containing the open position
            return [] # No open position found for this symbol
        except Exception as e:
            print(f"Error getting open positions: {e}")
            return []
    
    async def start_ticker_websocket(self, symbols):
        """Start a websocket connection to get real-time ticker data"""
        if not isinstance(symbols, list):
            symbols = [symbols]
            
        # Format symbols for Binance websocket (lowercase and with @ prefix)
        formatted_symbols = [f"{symbol.lower()}@ticker" for symbol in symbols]
        
        # Binance websocket URL
        url = "wss://stream.binance.com:9443/ws/" + "/".join(formatted_symbols)
        
        import websockets
        
        self.websocket_running = True
        
        try:
            async with websockets.connect(url) as websocket:
                print(f"WebSocket connected for {', '.join(symbols)}")
                
                while self.websocket_running:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    # Extract symbol and price
                    symbol = data['s']  # Symbol
                    price = float(data['c'])  # Current price
                    
                    # Update latest price
                    self.latest_prices[symbol] = price
                    
                    # You can add custom processing here
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
            self.websocket_running = False
    
    def get_latest_websocket_price(self, symbol):
        """Get the latest price from websocket data"""
        if self.mock_mode:
            # In mock mode, use the same price as get_latest_price
            return self.get_latest_price(symbol)
        
        return self.latest_prices.get(symbol)
    
    def stop_websocket(self):
        """Stop the websocket connection"""
        self.websocket_running = False