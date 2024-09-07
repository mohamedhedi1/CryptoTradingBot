import ccxt
import pandas as pd
import numpy as np

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

symbol = 'BTC/USDT'
timeframe = '1h'


# Fetch historical data
def fetch_ohlcv(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ALMA calculation
def alma(src, length, offset, sigma):
    m = (length - 1) / 2
    s = length / sigma
    w = np.exp(-((np.arange(length) - m) ** 2) / (2 * s ** 2))
    w /= np.sum(w)
    convolved = np.convolve(src, w, mode='valid')
    
    # Pad the result to match the length of the input
    pad_width = len(src) - len(convolved)
    return np.pad(convolved, (pad_width, 0), mode='constant', constant_values=np.nan)

# ATR calculation
def atr(df, period=14):
    df['tr'] = np.maximum((df['high'] - df['low']).abs(), 
                          np.maximum((df['high'] - df['close'].shift(1)).abs(), 
                                     (df['low'] - df['close'].shift(1)).abs()))
    return df['tr'].rolling(window=period).mean()

# Supertrend calculation
def supertrend(df, length, multiplier):
    df['atr'] = atr(df, period=length)
    df['hpf'] = df[['high', 'low', 'close']].mean(axis=1)
    df['upperBand'] = df['hpf'] + multiplier * df['atr']
    df['lowerBand'] = df['hpf'] - multiplier * df['atr']
    df['isAboveSupertrend'] = df['close'] > df['upperBand']
    df['isBelowSupertrend'] = df['close'] < df['lowerBand']

# Fetch data and apply strategy
df = fetch_ohlcv(symbol, timeframe)

print(df)

# Calculate ALMA
df['fastALMA'] = alma(df['close'], 20, 0.85, 6)
df['slowALMA'] = alma(df['close'], 50, 0.85, 6)

# Trend and momentum detection
df['isUpTrend'] = np.where(df['fastALMA'] > df['slowALMA'], 1, 0)
df['isDownTrend'] = np.where(df['fastALMA'] < df['slowALMA'], 1, 0)
df['fako'] = df['close'] - df['close'].shift(50)
df['isPositiveMomentum'] = df['fako'] > 0

# Supertrend
supertrend(df, 10, 3.0)

# Signals
df['isBuySignal'] = (df['isUpTrend'] == 1) & (df['isPositiveMomentum'] == 1)
df['isSellSignal'] = (df['close'] < df['close'].rolling(165).mean()) & (df['fako'] < -0.0163)

# Function to place orders
def place_order(symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f'Order placed: {order}')
    except Exception as e:
        print(f'An error occurred: {e}')

# Execute strategy
for i in range(1, len(df)):
    if df['isBuySignal'].iloc[i] and not df['isBuySignal'].iloc[i - 1]:
        place_order(symbol, 'buy', 0.0001)  # Adjust amount as needed
        print("i'm buying now")

    if df['isSellSignal'].iloc[i] and not df['isSellSignal'].iloc[i - 1]:
        place_order(symbol, 'sell', 0.0001)  # Adjust amount as needed
        print("i'm selling now")
