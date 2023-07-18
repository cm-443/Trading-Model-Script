import json
from datetime import datetime, timedelta
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import csv
import requests
import os
import pathlib


# OANDA API account details
access_token = 'xxxxxxxxxxxxxxxxx'
accountID = 'xxxxxxxxxxxxxxxxxx'

directory_path = './csv'

# Initialize the API client
client = oandapyV20.API(access_token=access_token, environment="live")

# Define the currency pair
currency_pair = 'USD_JPY'

# Define the start and end dates for the historical pricing data
start_date = datetime(2023, 6, 16)  # YYYY, MM, DD
end_date = start_date + timedelta(days=1) # Add one day to the start date to get the end date for one day

# Define the subinterval duration (in seconds)
subinterval_duration = 300

# Create a list to keep track of the CSV files that are created
csv_files = []

# Loop over the subintervals within the specified day
current_date = start_date
while current_date < end_date:
    subinterval_start = current_date
    subinterval_end = subinterval_start + timedelta(seconds=subinterval_duration)

    # Define the request parameters for the historical pricing data
    params = {
        'granularity': 'S5',
        'from': subinterval_start.isoformat('T') + 'Z',
        'to': subinterval_end.isoformat('T') + 'Z'
    }

    # Define the request endpoint
    endpoint = instruments.InstrumentsCandles(instrument=currency_pair, params=params)

    # Execute the request and retrieve the response
    response = client.request(endpoint)

    # Parse the response and extract the pricing data
    if response['candles']:
        prices = []
        for candle in response['candles']:
            prices.append({
                'timestamp': datetime.fromisoformat(candle['time'][:-4]),
                'open': candle['mid']['o'],
                'high': candle['mid']['h'],
                'low': candle['mid']['l'],
                'close': candle['mid']['c']
            })

        # Create a CSV file name based on the start and end dates
        csv_file_name = f'{directory_path}/{subinterval_start.strftime("%Y%m%d_%H%M%S")}.csv'

        # Write the results to a CSV file
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close'])

            # Create a list of timestamps for each 5-second interval within the subinterval
            interval_timestamps = [subinterval_start + timedelta(seconds=i) for i in range(0, subinterval_duration, 5)]

            # Loop over the interval timestamps and write the corresponding pricing data to the CSV file
            for timestamp in interval_timestamps:
                # Find the first price in the response that corresponds to the current interval timestamp
                index = next((i for i, price in enumerate(prices) if price['timestamp'] >= timestamp), None)

                if index is not None:
                    writer.writerow(
                        [prices[index]['timestamp'], prices[index]['open'], prices[index]['high'], prices[index]['low'],
                         prices[index]['close']])

            # Add the
                        # Add the CSV file name to the list
            csv_files.append(os.path.basename(csv_file_name))

            print(f'Results written to {csv_file_name} file.')

    current_date = subinterval_end

# Print the list of CSV files that were created outside the while loop.
now = datetime.now()

# convert list to string
list_str = f"\nlist_{now.strftime('%Y%m%d_%H%M%S')} = " + str(csv_files)

# open the file in append mode
with open('lists.py', 'a') as f:
    f.write(list_str)


