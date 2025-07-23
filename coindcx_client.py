import os
import time
import config
import pandas as pd
import requests
import json
import hmac
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('coindcx_client')

class CoinDCXClient:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        self.connection_failed = False
        
        # CoinDCX API endpoints
        self.base_url = "https://api.coindcx.com"
        self.public_url = "https://public.coindcx.com"
        
        # Initialize CoinDCX client
        api_key = os.getenv('COINDCX_API_KEY')
        api_secret = os.getenv('COINDCX_API_SECRET')
        
        if not api_key or not api_secret:
            self.logger.error("CoinDCX API credentials not found in environment variables")
            raise ValueError("CoinDCX API credentials not found in environment variables")
            
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Test connection
        try:
            self.logger.info("🔗 Testing CoinDCX connection...")
            self.get_server_time()
            self.logger.info("✅ CoinDCX connection successful!")
        except Exception as e:
            self.logger.error(f"❌ CoinDCX connection failed: {e}")
            raise ConnectionError(f"Failed to connect to CoinDCX API: {e}")

    def _generate_signature(self, secret_bytes, body):
        """Generate HMAC signature for CoinDCX API"""
        return hmac.new(secret_bytes, body, hashlib.sha256).hexdigest()

    def _make_authenticated_request(self, method, endpoint, params=None, data=None):
        """Make authenticated request to CoinDCX API"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key
        }
        
        if data:
            body = json.dumps(data, separators=(',', ':'))
        else:
            body = ""
        
        secret_bytes = bytes(self.api_secret, encoding='utf-8')
        body_bytes = bytes(body, encoding='utf-8')
        signature = self._generate_signature(secret_bytes, body_bytes)
        headers['X-AUTH-SIGNATURE'] = signature
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"❌ CoinDCX API request failed: {e}")
            raise

    def get_server_time(self):
        """Get server time"""
        response = requests.get(f"{self.public_url}/market_data/current_prices")
        return int(time.time() * 1000)

    def get_account_balance(self):
        """Get account balance"""
        try:
            endpoint = "/exchange/v1/users/balances"
            response = self._make_authenticated_request("POST", endpoint, {})
            
            if response:
                balances = {}
                for balance in response:
                    currency = balance.get('currency', '')
                    available = float(balance.get('balance', 0))
                    if available > 0:
                        balances[currency] = available
                
                # Focus on INR balance for spot trading
                inr_balance = balances.get('INR', 0.0)
                self.logger.info(f"💰 INR Balance: {inr_balance}")
                return balances
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance: {e}")
            return {}

    def get_available_symbols(self):
        """Get list of available trading symbols on CoinDCX"""
        try:
            # Try the current prices endpoint first
            response = requests.get(f"{self.public_url}/market_data/current_prices", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            if isinstance(data, dict):
                # Handle dict format
                for key, value in data.items():
                    symbols.append(key)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'market' in item:
                        symbols.append(item['market'])
            
            return sorted(symbols)
        except Exception as e:
            self.logger.error(f"❌ Error getting available symbols: {e}")
            return []

    def validate_symbol(self, symbol):
        """Check if a symbol is available for trading"""
        try:
            price = self.get_current_price(symbol)
            return price is not None
        except Exception as e:
            self.logger.error(f"❌ Error validating symbol {symbol}: {e}")
            return False

    def get_symbol_info(self, symbol):
        """Get symbol information"""
        if not self.validate_symbol(symbol):
            self.logger.warning(f"⚠️ Symbol {symbol} is not available for trading")
            return None
        
        try:
            response = requests.get(f"{self.public_url}/market_data/trade_details")
            for market in response:
                if market['symbol'] == symbol:
                    return {
                        'symbol': market['symbol'],
                        'base_currency': market['base_currency_short_name'],
                        'target_currency': market['target_currency_short_name'],
                        'min_quantity': float(market['min_quantity']),
                        'max_quantity': float(market['max_quantity']),
                        'min_price': float(market['min_price']),
                        'max_price': float(market['max_price']),
                        'step_size': float(market['step']),
                        'tick_size': float(market['min_price'])
                    }
            return None
        except Exception as e:
            self.logger.error(f"❌ Error getting symbol info: {e}")
            return None

    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            response = requests.get(f"{self.public_url}/market_data/current_prices", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Handle dictionary format: {'DOGEINR': '0.12345', 'BTCINR': '5000000', ...}
            if isinstance(data, dict):
                price_str = data.get(symbol)
                if price_str is not None:
                    return float(price_str)
                else:
                    self.logger.warning(f"❌ Symbol {symbol} not found in price data")
                    return None
            
            # Handle list format (fallback)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get('market') == symbol:
                        return float(item.get('last_price', 0))
            
            self.logger.error(f"❌ Unexpected data format for current price: {type(data)}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error getting current price: {e}")
            return None

    def get_candlestick_data(self, symbol, interval='1m', limit=100):
        """Get candlestick data for technical analysis"""
        try:
            # Map interval to CoinDCX format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1w',
                '1M': '1M'
            }
            
            # Default to '1h' if interval not in map
            interval = interval_map.get(interval, '1h')
            
            # Convert symbol to CoinDCX format (e.g., 'DOGEINR' -> 'I-DOGE_INR')
            if symbol.endswith('INR'):
                base_currency = symbol.replace('INR', '')
                pair = f'I-{base_currency}_INR'
            else:
                pair = symbol  # Fallback to original symbol if not INR pair
            
            # CoinDCX public candlestick endpoint
            url = f"{self.public_url}/market_data/candles"
            params = {
                'pair': pair,
                'interval': interval,
                'limit': min(limit, 1000)  # API might have a limit
            }
            
            self.logger.debug(f"Fetching candlestick data for {pair} with interval {interval}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                self.logger.error(f"❌ Invalid response format for {pair}. Expected list, got {type(data)}. Response: {data}")
                return None
                
            if not data:
                self.logger.warning(f"⚠️ Empty response received for {pair}. This pair may not be supported or no data available.")
                return None
                
            # Convert to expected format: [timestamp, open, high, low, close, volume]
            candles = []
            for i, candle in enumerate(data):
                try:
                    if isinstance(candle, dict):
                        # Handle dictionary format
                        candles.append([
                            int(candle.get('time', 0)),  # timestamp (in ms)
                            float(candle.get('open', 0)),
                            float(candle.get('high', 0)),
                            float(candle.get('low', 0)),
                            float(candle.get('close', 0)),
                            float(candle.get('volume', 0))
                        ])
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        # Handle list/tuple format
                        candles.append([
                            int(candle[0]),    # timestamp (in ms)
                            float(candle[1]),  # open
                            float(candle[2]),  # high
                            float(candle[3]),  # low
                            float(candle[4]),  # close
                            float(candle[5])   # volume
                        ])
                    else:
                        self.logger.warning(f"⚠️ Unexpected candle format at index {i}: {candle}")
                except (ValueError, IndexError, AttributeError) as e:
                    self.logger.warning(f"⚠️ Error parsing candle data at index {i}: {e}. Candle: {candle}")
            
            if not candles:
                self.logger.error(f"❌ No valid candle data received for {pair} after parsing. Raw data: {data}")
                return None
                
            # Sort candles by timestamp in ascending order (oldest first)
            candles.sort(key=lambda x: x[0])
            
            self.logger.info(f"✅ Fetched {len(candles)} candles for {pair} (interval: {interval})")
            return candles
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error fetching candlestick data for {symbol}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                self.logger.error(f"Response status: {e.response.status_code}")
                self.logger.error(f"Response body: {e.response.text}")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in get_candlestick_data: {str(e)}", exc_info=True)
            
        return None

    def format_candlestick_data(self, candles):
        """Format candlestick data into pandas DataFrame"""
        try:
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime - handle both seconds and milliseconds
            # Check if timestamps are in milliseconds (typical for APIs) or seconds
            sample_ts = df['timestamp'].iloc[0] if len(df) > 0 else 0
            unit = 'ms' if sample_ts > 1_000_000_000_000 else 's'  # If timestamp > year 2001, assume ms
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit)
            df.set_index('timestamp', inplace=True)
            
            # Ensure all price columns are float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting candlestick data: {e}")
            return pd.DataFrame()

    def place_order(self, symbol, side, order_type, quantity, price=None):
        """Place a spot order"""
        try:
            order_data = {
                'side': side.lower(),  # 'buy' or 'sell'
                'order_type': order_type.lower(),  # 'limit' or 'market'
                'market': symbol,
                'quantity': str(quantity)
            }
            
            if order_type.lower() == 'limit' and price:
                order_data['price'] = str(price)
            
            response = self._make_authenticated_request('POST', '/exchange/v1/orders/create', data=order_data)
            return response
        except Exception as e:
            self.logger.error(f"❌ Error placing order: {e}")
            return None

    def get_order_status(self, order_id):
        """Get order status"""
        try:
            data = {'id': order_id}
            response = self._make_authenticated_request('POST', '/exchange/v1/orders/status', data=data)
            return response
        except Exception as e:
            self.logger.error(f"❌ Error getting order status: {e}")
            return None

    def cancel_order(self, order_id):
        """Cancel an order"""
        try:
            data = {'id': order_id}
            response = self._make_authenticated_request('POST', '/exchange/v1/orders/cancel', data=data)
            return response
        except Exception as e:
            self.logger.error(f"❌ Error cancelling order: {e}")
            return None

    def get_open_orders(self, symbol=None):
        """Get open orders with consistent formatting"""
        try:
            endpoint = "/exchange/v1/orders/active_orders"
            params = {}
            if symbol:
                params["market"] = symbol
                
            response = self._make_authenticated_request("POST", endpoint, params=params)
            
            # Handle the case where response is a dict with 'orders' key
            if isinstance(response, dict) and 'orders' in response:
                orders = response['orders']
                if not isinstance(orders, list):
                    self.logger.warning(f"Unexpected orders format: {orders}")
                    return []
                response = orders
            
            # Ensure response is a list of dictionaries with expected keys
            if not isinstance(response, list):
                self.logger.error(f"Unexpected response format from get_open_orders: {response}")
                return []
                
            # Format each order to ensure consistent structure
            formatted_orders = []
            for order in response:
                if not isinstance(order, dict):
                    self.logger.warning(f"Skipping malformed order: {order}")
                    continue
                    
                formatted_order = {
                    'id': str(order.get('id', '')),
                    'symbol': order.get('market', ''),
                    'side': order.get('side', '').lower(),  # Ensure lowercase for consistency
                    'price': str(order.get('price', '0')),
                    'quantity': str(order.get('quantity', '0')),
                    'status': order.get('status', '').lower(),
                    'type': order.get('type', '').lower(),
                    'timestamp': order.get('created_at', '')
                }
                formatted_orders.append(formatted_order)
                
            self.logger.debug(f"Formatted {len(formatted_orders)} open orders")
            return formatted_orders
            
        except Exception as e:
            self.logger.error(f"❌ Error getting open orders: {e}")
            return []

    def get_order_history(self, symbol=None, limit=5):
        """Get recent order history"""
        try:
            # CoinDCX uses different endpoint for order history
            endpoint = "/exchange/v1/orders/active_orders_count"
            params = {}
            if symbol:
                params["market"] = symbol
                
            response = self._make_authenticated_request("POST", endpoint, params)
            
            # If that fails, try alternative endpoint
            if not response:
                endpoint = "/exchange/v1/orders/trade_history"
                response = self._make_authenticated_request("POST", endpoint, params)
            
            return response if response else []
        except Exception as e:
            self.logger.error(f"❌ Error getting order history: {e}")
            return []
