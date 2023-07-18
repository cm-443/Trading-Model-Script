import pandas as pd
import numpy as np
import ta
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error


master_list = ['20230501_000000.csv', '20230501_000500.csv']
master_list = ['csv/' + filename for filename in master_list]

def bollinger_bands(data, window, num_std_dev):
    sma = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def rsi_divergence(data, rsi_data, window):
    data_diff = data.diff(window)
    rsi_diff = rsi_data.diff(window)
    divergence = np.where(data_diff * rsi_diff < 0, 1, 0)
    return divergence

def calculate_tdi(df, rsi_period=13, bb_period=34, ema_period=7):
    # Calculate RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], rsi_period).rsi()
    # Calculate Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(df['close'], bb_period)
    df['bb_high'], df['bb_mid'], df['bb_low'] = bb_indicator.bollinger_hband(), bb_indicator.bollinger_mavg(), bb_indicator.bollinger_lband()
    # Calculate EMA of RSI
    df['ema_rsi'] = df['rsi'].ewm(span=ema_period).mean()
    # Calculate TDI
    df['tdi'] = (df['rsi'] + df['bb_mid'] + df['ema_rsi']) / 3
    return df

dfs = []
for csv_file in master_list:
    df = pd.read_csv(csv_file, index_col='timestamp', usecols=['timestamp', 'open', 'high', 'low', 'close'])
    dfs.append(df)
df = pd.concat(dfs)
df = df.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
df['normalized_price'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))

window = 14
num_std_dev = 2
df['upper_band'], df['lower_band'] = bollinger_bands(df['close'], window, num_std_dev)

df = calculate_tdi(df)  # compute TDI

df['rsi_divergence'] = rsi_divergence(df['close'], df['rsi'], window)

df.dropna(inplace=True)

X = []
y = []
window_size = 14
for i in range(len(df) - window_size):
    X.append(df[['normalized_price', 'upper_band', 'lower_band', 'rsi', 'rsi_divergence', 'tdi']].iloc[i:i + window_size].values)
    y.append(df['normalized_price'].iloc[i + window_size])
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

model = Sequential()
model.add(LSTM(100, activation='tanh', input_shape=(window_size, 6), return_sequences=True)) # adjust input shape for 6 features
model.add(LSTM(100, activation='tanh', return_sequences=True))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', run_eagerly=True)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[es, mc])

# Load the best saved model
model = tf.keras.models.load_model('best_model.h5')

# Evaluate the model on the testing set
loss = model.evaluate(X_test, y_test)
print('Test Loss:', loss)

# Generate predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Save the model
model.save('RSI_Divergence.h5')

