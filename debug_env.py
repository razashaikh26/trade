#!/usr/bin/env python3
import os
from dotenv import load_dotenv

print("=== ENVIRONMENT VARIABLES DEBUG ===")

# Load .env file
print("1. Loading .env file...")
load_dotenv()
print("   ✅ .env file loaded")

# Check if .env file exists
env_file_exists = os.path.exists('.env')
print(f"2. .env file exists: {env_file_exists}")

# Check environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

print(f"3. BINANCE_API_KEY found: {'✅ YES' if api_key else '❌ NO'}")
print(f"4. BINANCE_API_SECRET found: {'✅ YES' if api_secret else '❌ NO'}")

if api_key:
    print(f"5. API Key length: {len(api_key)} characters")
    print(f"6. API Key starts with: {api_key[:8]}...")
else:
    print("5. ❌ API Key is None or empty")

if api_secret:
    print(f"7. API Secret length: {len(api_secret)} characters")
    print(f"8. API Secret starts with: {api_secret[:8]}...")
else:
    print("7. ❌ API Secret is None or empty")

# Check all environment variables that start with BINANCE
print("\n9. All BINANCE-related environment variables:")
for key, value in os.environ.items():
    if 'BINANCE' in key.upper():
        print(f"   {key}: {'✅ SET' if value else '❌ EMPTY'}")

print("\n=== TROUBLESHOOTING TIPS ===")
print("If API credentials are not found:")
print("1. Make sure .env file is in the same directory as this script")
print("2. Check .env file format (no spaces around =)")
print("3. Make sure there are no quotes around the values")
print("4. Example format:")
print("   BINANCE_API_KEY=your_key_here")
print("   BINANCE_API_SECRET=your_secret_here")
