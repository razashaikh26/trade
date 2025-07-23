import os
import time
import config
import pandas as pd
import ccxt
import asyncio
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
import logging

class BinanceClient:
    def __init__(self, mock_mode=False, testnet=False):
        # Load environment variables
        load_dotenv()
        
        self.mock_mode = mock_mode
        self.testnet = testnet
        self.connection_failed = False
        
        # Mock mode settings
        if mock_mode:
            self.mock_balance = {'USDT': 1000.0}
            self.mock_positions = []
            self.mock_orders = []
            print(" BinanceClient initialized in MOCK MODE")
            return
        
        # Initialize Binance client using CCXT
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print(" API credentials not found. Running in mock mode.")
            self.mock_mode = True
            self.mock_balance = {'USDT': 1000.0}
            self.mock_positions = []
            self.mock_orders = []
            return
            
        # Try multiple connection methods (fast alternatives)
        connection_successful = False
        
        # Method 1: Standard connection
        try:
            print(" Attempting standard Binance connection...")
            self.client = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'timeout': 10000,
            })

            # Set testnet mode if enabled
            if testnet:
                self.client.set_sandbox_mode(True)
                print(" BinanceClient initialized in TESTNET MODE")
            
            # Test connection
            self._test_connection()
            connection_successful = True
            print(" Standard connection successful!")
            
        except Exception as e:
            print(f" Standard connection failed: {e}")
            
        # Method 2: Alternative endpoints (faster than proxies)
        if not connection_successful:
            print(" Trying alternative Binance endpoints...")
            alternative_urls = [
                'https://api1.binance.com',
                'https://api2.binance.com', 
                'https://api3.binance.com'
            ]
            
            for alt_url in alternative_urls:
                try:
                    print(f" Testing endpoint: {alt_url}")
                    self.client = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                        'options': {'defaultType': 'future'},
                        'timeout': 8000,
                        'urls': {
                            'api': {
                                'public': alt_url + '/api',
                                'private': alt_url + '/api',
                            }
                        }
                    })
                    
                    # Set testnet mode if enabled
                    if testnet:
                        self.client.set_sandbox_mode(True)
                    
                    # Test the alternative endpoint connection
                    self._test_connection()
                    connection_successful = True
                    print(f" Alternative endpoint successful: {alt_url}")
                    break
                    
                except Exception as e:
                    print(f" Endpoint {alt_url} failed: {str(e)[:50]}...")
                    continue
        
        # If all methods fail, fall back to mock mode
        if not connection_successful:
            print(" All connection methods failed!")
            print(" Falling back to MOCK MODE")
            self.mock_mode = True
            self.connection_failed = True
            self.mock_balance = {'USDT': 1000.0}
            self.mock_positions = []
            self.mock_orders = []
        
        # Store latest prices
        self.latest_prices = {}
        self.websocket_running = False
        
        # Store latest prices
        self.latest_prices = {}
        self.websocket_running = False
        
    def _test_connection(self):
        """Test connection to Binance API"""
        if self.mock_mode:
            return True
            
        try:
            # Try to ping the server
            self.client.load_markets()
            print(" Successfully connected to Binance API")
            return True
        except ccxt.NetworkError as e:
            if "restricted location" in str(e).lower():
                print(" Geographic restriction detected - Binance API blocked")
                raise Exception("Geographic restriction")
            else:
                print(f" Network error: {e}")
                raise
        except Exception as e:
            print(f" Connection test failed: {e}")
            raise
    
    def get_account_balance(self, asset='USDT'):
        """Get account balance for a specific asset from both Futures and Spot wallets"""
        if self.mock_mode:
            # Return mock balance if in mock mode
            return self.mock_balance.get(asset, 0.0)
        
        try:
            # First check Futures account balance
            futures_balance = self.client.fetch_balance()
            
            # Check for balance in futures account first
            common_currencies = ['USDT', 'BUSD', 'BNB', 'USDC', 'FDUSD', 'ETH', 'BTC', 'LDUSDT', 'BFUSD']
            
            # Check futures balance
            for currency in [asset] + common_currencies:
                if currency in futures_balance and isinstance(futures_balance[currency], dict):
                    total_balance = futures_balance[currency].get('total', 0)
                    free_balance = futures_balance[currency].get('free', 0)
                    if total_balance > 0 or free_balance > 0:
                        available = free_balance if free_balance > 0 else total_balance
                        return float(available)
            
            # If no futures balance found, check spot wallet
            
            # Temporarily switch to spot mode to check spot balance
            original_type = self.client.options.get('defaultType', 'future')
            self.client.options['defaultType'] = 'spot'
            
            try:
                spot_balance = self.client.fetch_balance()
                
                # Check spot balance - first try common currencies, then any non-zero balance
                for currency in [asset] + common_currencies:
                    if currency in spot_balance and isinstance(spot_balance[currency], dict):
                        total_balance = spot_balance[currency].get('total', 0)
                        free_balance = spot_balance[currency].get('free', 0)
                        if total_balance > 0 or free_balance > 0:
                            available = free_balance if free_balance > 0 else total_balance
                            return float(available)
                
                # If no common currencies found, check for ANY non-zero balance
                for currency, balance_info in spot_balance.items():
                    if isinstance(balance_info, dict):
                        total_balance = balance_info.get('total', 0)
                        free_balance = balance_info.get('free', 0)
                        if total_balance > 0 or free_balance > 0:
                            available = free_balance if free_balance > 0 else total_balance
                            return float(available)
                
            finally:
                # Always restore original type
                self.client.options['defaultType'] = original_type
            
            return 0.0
                
        except Exception as e:
            print(f"CRITICAL ERROR: Could not get account balance. Check API keys and permissions. Error: {e}")
            return 0.0
    
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
    
    def place_order(self, symbol, side, quantity, order_type='market', price=None, stop_loss=None, take_profit=None, reduce_only=False):
        """Place an order on Binance with optional SL/TP for futures"""
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
                'info': {'mock': True, 'stopLoss': stop_loss, 'takeProfit': take_profit, 'reduceOnly': reduce_only}
            }
            
            # Simulate balance change
            cost = quantity * price
            if side.lower() == 'buy':
                if reduce_only:
                    # Closing a short position
                    self.mock_balance['USDT'] += cost
                else:
                    # Opening a long position
                    self.mock_balance['USDT'] -= cost
            else:  # sell
                if reduce_only:
                    # Closing a long position
                    self.mock_balance['USDT'] += cost
                else:
                    # Opening a short position
                    self.mock_balance['USDT'] -= cost
            
            return mock_order

        # Real mode for futures trading
        try:
            params = {
                'reduceOnly': reduce_only  # CRITICAL: Use reduce_only parameter to close positions
            }

            # For futures, we can often set SL/TP in the same order request
            # The exact parameter names can vary, 'stopPrice' and 'takeProfitPrice' are common
            if stop_loss:
                params['stopPrice'] = stop_loss
                params['takeProfitPrice'] = take_profit # Binance often requires both or neither

            # Enhanced logging for debugging
            print(f"🔄 Attempting to place order: {side} {quantity} {symbol} (reduce_only={reduce_only})")
            print(f"   Order type: {order_type}, Price: {price}, Params: {params}")
            if self.testnet:
                print(f"   🧪 TESTNET MODE: Order will be placed on testnet")
            
            # Check if we have an active connection
            if hasattr(self, 'connection_failed') and self.connection_failed:
                print(f"❌ Cannot place order: Connection to Binance failed")
                return None

            order = self.client.create_order(
                symbol=symbol,
                type=order_type, 
                side=side,
                amount=quantity,
                price=price, # For limit orders
                params=params
            )

            print(f"✅ Order placed successfully: {order.get('id', 'N/A')}")
            return order
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error placing order: {error_msg}")
            print(f"   Symbol: {symbol}, Side: {side}, Quantity: {quantity}, Type: {order_type}")
            print(f"   Reduce Only: {reduce_only}, Params: {params}")
            
            # Enhanced error analysis
            if "insufficient" in error_msg.lower():
                print(f"💡 DIAGNOSIS: Insufficient balance or position size issue")
            elif "permission" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print(f"💡 DIAGNOSIS: API permission issue - check testnet API keys")
            elif "symbol" in error_msg.lower():
                print(f"💡 DIAGNOSIS: Symbol format issue - check if {symbol} is valid for testnet")
            elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
                print(f"💡 DIAGNOSIS: Network connectivity issue")
            else:
                print(f"💡 DIAGNOSIS: Unknown error - full details: {error_msg}")
            
            # Attempt to place a simple order if the complex one fails
            try:
                print("🔄 Attempting to place order without SL/TP...")
                simple_params = {'reduceOnly': reduce_only} if reduce_only else {}
                
                order = self.client.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=quantity,
                    price=price,
                    params=simple_params
                )
                print("✅ Simple order placed successfully. You must set SL/TP manually.")
                return order
            except Exception as e2:
                error_msg2 = str(e2)
                print(f"❌ Failed to place even a simple order: {error_msg2}")
                print(f"   This might be a connection issue, insufficient balance, or API restriction")
                
                # Try one more time with minimal parameters for market orders
                if order_type.lower() == 'market':
                    try:
                        print("🔄 Final attempt with minimal market order...")
                        minimal_order = self.client.create_market_order(
                            symbol=symbol,
                            side=side,
                            amount=quantity,
                            params={'reduceOnly': reduce_only} if reduce_only else {}
                        )
                        print("✅ Minimal market order placed successfully!")
                        return minimal_order
                    except Exception as e3:
                        error_msg3 = str(e3)
                        print(f"❌ All order attempts failed. Final error: {error_msg3}")
                        
                        # Critical debugging for testnet
                        if self.testnet:
                            print(f"🧪 TESTNET DEBUGGING:")
                            print(f"   - Check if testnet API keys are correctly set")
                            print(f"   - Verify testnet has sufficient balance")
                            print(f"   - Confirm symbol {symbol} exists on testnet")
                            print(f"   - Check if position actually exists to close")
                        
                        return None
                return None
    
    def get_income_history(self, symbol: str, start_time: int) -> List[Dict]:
        """Get income history (realized PNL) for a symbol since a specific time."""
        if self.mock_mode:
            # In mock mode, we don't have a real trade history
            return []
        
        try:
            # Fetch income history using the explicit private API method for futures
            income_history = self.client.fapiPrivateGetIncome(
                params={
                    'symbol': symbol,
                    'incomeType': 'REALIZED_PNL',
                    'startTime': start_time,
                    'limit': 1000
                }
            )
            return income_history
        except Exception as e:
            print(f"Error getting income history for {symbol}: {e}")
            return []

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

    def get_recent_trades(self, symbol: str, limit: int = 100, since: int = None) -> List[Dict]:
        """
        Fetch recent trades for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'DOGEUSDT')
            limit: Maximum number of trades to return (default: 100, max: 1000)
            since: Timestamp in milliseconds to start fetching trades from
            
        Returns:
            List of trade dictionaries with trade details
        """
        if self.mock_mode:
            # Return mock trade data for testing
            mock_trade = {
                'id': 'mock-trade-1',
                'symbol': symbol,
                'orderId': 'mock-order-1',
                'side': 'buy',
                'price': str(self.get_latest_price(symbol) or 0.0),
                'qty': '100',
                'realizedPnl': '0.0',
                'marginAsset': 'USDT',
                'quoteQty': '0.0',
                'commission': '0.0',
                'commissionAsset': 'USDT',
                'time': int(time.time() * 1000),
                'positionSide': 'LONG',
                'buyer': True,
                'maker': False
            }
            return [mock_trade] * min(limit, 10)  # Return up to 10 mock trades
        
        try:
            # Ensure symbol is in correct format (no slashes)
            symbol = symbol.replace('/', '')
            
            # Prepare parameters
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            if since is not None:
                params['startTime'] = since
            
            # Fetch trades
            trades = self.client.fapiPrivateGetUserTrades(params)
            
            # Format trades consistently
            formatted_trades = []
            for trade in trades:
                formatted_trade = {
                    'id': str(trade.get('id', '')),
                    'symbol': trade.get('symbol', symbol),
                    'orderId': str(trade.get('orderId', '')),
                    'side': trade.get('side', '').lower(),
                    'price': trade.get('price', '0'),
                    'qty': trade.get('qty', '0'),
                    'realizedPnl': trade.get('realizedPnl', '0'),
                    'marginAsset': trade.get('marginAsset', 'USDT'),
                    'quoteQty': trade.get('quoteQty', '0'),
                    'commission': trade.get('commission', '0'),
                    'commissionAsset': trade.get('commissionAsset', 'USDT'),
                    'time': int(trade.get('time', time.time() * 1000)),
                    'positionSide': trade.get('positionSide', 'BOTH'),
                    'buyer': trade.get('isBuyer', False),
                    'maker': trade.get('isMaker', False)
                }
                formatted_trades.append(formatted_trade)
            
            # Sort by time (oldest first)
            formatted_trades.sort(key=lambda x: x['time'])
            
            # Apply limit after formatting
            return formatted_trades[-limit:] if limit else formatted_trades
            
        except Exception as e:
            error_msg = f"Error fetching recent trades: {str(e)}"
            print(f"❌ {error_msg}")
            logging.error(error_msg, exc_info=True)
            return []