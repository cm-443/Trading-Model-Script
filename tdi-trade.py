import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
from collections import deque
import oandapyV20.types as types
import oandapyV20.types as tp
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades
import requests
import ta
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator

accountID = 'xxxxxxxxxxxxxxx'
access_token = 'xxxxxxxxxxxxxx'
client = oandapyV20.API(access_token=access_token, environment="live")

params = {
    "instruments": "USD_JPY",
}

model = load_model('RSI_Divergence.h5')
print("Model summary:")
model.summary()
close_prices = []



def calculate_tdi(prices, rsi_period=13, bb_period=34, ema_period=7):
    data = pd.DataFrame(prices, columns=['Close'])

    # Calculate RSI
    rsi_indicator = ta.momentum.RSIIndicator(data['Close'], rsi_period)
    data['rsi'] = rsi_indicator.rsi()
    print(f"RSI: {data['rsi']}")

    # Calculate Bollinger Bands
    # Calculate Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(data['Close'], bb_period)
    data['bb_high'], data['bb_mid'], data[
        'bb_low'] = bb_indicator.bollinger_hband(), bb_indicator.bollinger_mavg(), bb_indicator.bollinger_lband()
    print(f"BB High: {data['bb_high']}")
    print(f"BB Mid: {data['bb_mid']}")
    print(f"BB Low: {data['bb_low']}")

    # Calculate EMA
    data['ema'] = data['rsi'].ewm(span=ema_period).mean()
    print(f"EMA: {data['ema']}")

    # Remove rows containing NaN values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Calculate TDI
    data['tdi'] = (data['rsi'] + data['bb_mid'] + data['ema']) / 3
    print(f"TDI: {data['tdi']}")

    return data['tdi'].values, data['rsi'].values, data['ema'].values



def bollinger_bands(data, window, num_std_dev):
    data = np.squeeze(data)  # Extract 1-dimensional array from ndarray
    sma = pd.Series(data).rolling(window=window).mean()
    std_dev = pd.Series(data).rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    # Return numpy arrays instead of pandas series
    return upper_band.to_numpy(), lower_band.to_numpy()


def has_open_trade(instrument):
    params = {"instrument": instrument}
    r = trades.TradesList(accountID, params=params)
    open_trades = client.request(r)["trades"]
    return len(open_trades) > 0



def rsi_divergence(prices, rsi, window_size):
    price_diff = np.diff(prices)
    rsi_diff = np.diff(rsi)

    # Ensure that both arrays have the same shape
    if len(price_diff) > len(rsi_diff):
        rsi_diff = np.pad(rsi_diff, (len(price_diff) - len(rsi_diff), 0), 'constant', constant_values=(0,))
    elif len(price_diff) < len(rsi_diff):
        price_diff = np.pad(price_diff, (len(rsi_diff) - len(price_diff), 0), 'constant', constant_values=(0,))

    price_divergence = np.where((price_diff > 0) & (rsi_diff < 0), 1, 0)
    price_divergence = np.pad(price_divergence, (1, 0), 'constant', constant_values=(0,))

    divergence = np.zeros(window_size)
    divergence[-len(price_divergence) // 2:] = price_divergence[::2]

    return divergence


def pad_arrays(*arrays, length):
    padded_arrays = []
    for arr in arrays:
        if len(arr) < length:
            pad_length = length - len(arr)
            padded_arr = np.pad(arr, (pad_length, 0), 'constant', constant_values=(0,))
        else:
            padded_arr = arr[-length:]  # Keep the last 'length' elements
        padded_arrays.append(padded_arr)

    return padded_arrays





def close_trade(trade_id):
    close_request = trades.TradeClose(accountID, trade_id)
    client.request(close_request)
    print(f"Trade {trade_id} closed")


def pricing_streaming():
    global params

    while True:
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))

            params = {
                "instruments": "USD_JPY",
                "Authorization": f"Bearer {access_token}"
            }

            r = pricing.PricingStream(accountID=accountID, params=params)
            rv = client.request(r)

            # window_size = 14
            window_size = 34
            close_prices = deque(maxlen=window_size + 34)

            for ticks in rv:
                if ticks['type'] == 'PRICE':
                    close_price = ticks['bids'][0]['price']
                    print(f"Latest close price for {params['instruments']}: {close_price}")
                    close_prices.append(float(close_price))
                    time.sleep(1)

                    if len(close_prices) >= window_size + 34:
                        close_prices_windowed = list(close_prices)[-window_size - 34:]
                        # Normalize the close price
                        close_prices_normalized = scaler.fit_transform(
                            np.array(close_prices_windowed[-34:]).reshape(-1,
                                                                          1))

                        # Calculate the RSI tdi

                        # Calculate the RSI tdi
                        tdi, rsi, ema = calculate_tdi(list(close_prices))
                        print("TDI Values: ", tdi)
                        print("RSI Values: ", rsi)
                        print("EMA Values: ", ema)

                        print(f"TDI Values length: {len(tdi)}")
                        print(f"RSI Values length: {len(rsi)}")
                        print(f"EMA Values length: {len(ema)}")

                        # Calculate tdi_div using rsi_divergence function
                        tdi_div = rsi_divergence(close_prices, tdi, window_size)
                        # padded_tdi_div, padded_rsi, padded_ema = pad_arrays(tdi_div, rsi, ema, length=window_size + 34)

                        # Pad the arrays
                        padded_tdi_div, padded_rsi, padded_ema = pad_arrays(tdi_div, rsi, ema, length=window_size + 34)
                        padded_tdi_div = padded_tdi_div[-34:]
                        padded_rsi = padded_rsi[-34:]

                        # Pad the arrays\
                        num_std_dev = 2

                        upper_band, lower_band = bollinger_bands(close_prices_normalized, window_size, num_std_dev)
                        # Convert to numpy ndarrays
                        upper_band = np.array(upper_band)
                        lower_band = np.array(lower_band)


                        # Slice the larger arrays to match the size of the smaller arrays
                        close_prices_normalized = close_prices_normalized[-34:]
                        upper_band = upper_band[-34:]
                        lower_band = lower_band[-34:]

                        print(f"Close Prices Normalized length: {len(close_prices_normalized)}")
                        print(f"Upper Band length: {len(upper_band)}")
                        print(f"Lower Band length: {len(lower_band)}")
                        print(f"Padded RSI length: {len(padded_rsi)}")
                        print(f"Padded TDI Div length: {len(padded_tdi_div)}")
                        print(f"TDI length: {len(tdi[-34:])}")

                        print("Shape of close_prices_normalized:", close_prices_normalized.shape)
                        print("Shape of upper_band:", upper_band.shape)
                        print("Shape of lower_band:", lower_band.shape)
                        print("Shape of padded_rsi:", padded_rsi.shape)
                        print("Shape of padded_tdi_div:", padded_tdi_div.shape)
                        print("Shape of tdi[-34:]:", tdi[-34:].shape)

                        # Stack the features
                        X = np.column_stack(
                            [close_prices_normalized, upper_band, lower_band, padded_rsi, padded_tdi_div, padded_ema]
                        )

                        print(f"Input data (X): {X}")
                        print(f"TDI: {tdi[-14:]}, RSI: {rsi[-14:]}, EMA: {ema[-14:]}, Divergence: {tdi_div[-14:]}")

                        # Make a prediction
                        X = X.reshape(1, window_size + 34, 5)

                        print("Shape of X before reshaping:", X.shape)

                        prediction = model.predict(X)[0][0]

                        print(
                            f"Prediction: {prediction}, Last Close Price Normalized: {close_prices_normalized[-1][0]}")

                        # Determine if the prediction indicates a buy or sell signal
                        buy_signal = prediction > close_prices_normalized[-1][0]
                        sell_signal = prediction < close_prices_normalized[-1][0]

                        # Check if there's already an open trade
                        open_trade = has_open_trade(params["instruments"])

                        # Retrieve price granularity and set take profit and stop loss distances
                        params_candles = {
                            "granularity": "S5"
                        }

                        r = instruments.InstrumentsCandles(instrument=params["instruments"], params=params_candles)
                        instrument_info = client.request(r)
                        price_granularity = len(str(instrument_info['candles'][0]['mid']['c']).split('.')[1])

                        take_profit_distance = 5 * (10 ** -(price_granularity - 1))
                        print(take_profit_distance)
                        stop_loss_distance = 5 * (10 ** -(price_granularity - 1))
                        units = 10

                        # Execute a buy order
                        if buy_signal and not open_trade:
                            print(f"Buy signal: {buy_signal}, Open trade: {open_trade}")
                            close_price = float(ticks['bids'][0]['price'])
                            tp_price = round(close_price + take_profit_distance, 3)
                            tp_price = tp.PriceValue(tp_price).value
                            sl_price = round(close_price - stop_loss_distance, 3)
                            sl_price = tp.PriceValue(sl_price).value

                            order = {
                                "order": {
                                    "instrument": params["instruments"],
                                    "units": str(units),
                                    "type": "MARKET",
                                    "positionFill": "DEFAULT",
                                    "side": "buy",
                                    "takeProfitOnFill": {
                                        "price": tp_price
                                    },
                                    "stopLossOnFill": {
                                        "price": sl_price
                                    }
                                }
                            }
                            order_request = orders.OrderCreate(accountID, data=order)
                            client.request(order_request)
                            print("Buy order executed")

                        # Execute a sell order
                        if sell_signal and not open_trade:
                            print(f"Sell signal: {sell_signal}, Open trade: {open_trade}")

                            close_price = float(ticks['asks'][0]['price'])
                            tp_price = round(close_price - take_profit_distance, 3)
                            tp_price = tp.PriceValue(tp_price).value
                            sl_price = round(close_price + stop_loss_distance, 3)
                            sl_price = tp.PriceValue(sl_price).value

                            order = {
                                "order": {
                                    "instrument": params["instruments"],
                                    "units": str(-units),
                                    "type": "MARKET",
                                    "positionFill": "DEFAULT",
                                    "side": "sell",
                                    "takeProfitOnFill": {
                                        "price": tp_price
                                    },
                                    "stopLossOnFill": {
                                        "price": sl_price
                                    }
                                }
                            }
                            order_request = orders.OrderCreate(accountID, data=order)
                            client.request(order_request)
                            print("Sell order executed")

        except requests.exceptions.ChunkedEncodingError:
            print("Connection broken (ChunkedEncodingError), attempting to reconnect...")
            time.sleep(5)
            continue
        except requests.exceptions.RequestException as e:
            print(f"Connection error ({type(e).__name__}): {e}, attempting to reconnect...")
            time.sleep(5)
            continue
        except ConnectionError:
            print("Connection error, attempting to reconnect...")
            time.sleep(5)
            continue
        except KeyboardInterrupt:
            print("Script stopped by the user.")
            break



if __name__ == "__main__":
    pricing_streaming()
