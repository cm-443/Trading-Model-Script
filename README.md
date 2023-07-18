# tdi
Model and trading script using the trader's dynamic index. 

These are currently configured for use with the Oanda API.
NOTE: RUNNING API KEYS EVEN LOCALLY IS DANGEROUS AND A REAL PRODUCTION SCRIPT SHOULD USE SECRETS MANAGER OR EQUIVALENT. 

To run them as is without modification, you must open an Oanda trading account and obtain an api key. No min deposit required, fees are built into spread. 

Instructions for the script as is with Oanda:

Update the accountID and access_token variables with your credentials on the 5-sec.py and tdi-trade.py

Gather the historical data with 5-sec.py script via the Oanda API. It gathers 5 sec data for a 24 hour period, you’ll need to update the date for each time you run it. (Theres probably a better way.) The script also requires a file directory in your IDE (PyCharm) named “csv” and and the script called “lists.py”. The script will run, upload the csv files into the csv directory, and print a list in the “list.py” file. If you need 30 days of data, you’ll update the date each time and a new list will be entered into the list.py

The script in list.py is suppose to compile all the lists into one master list and append the 'csv/' prefix when ran but its erroring out. Needs debugging. The model has a line for setting the prefix however.

Once you have a list, add it to the tdi-model.py and run it. When finished, it will output the model in the main directory.

After, run the trading script and update the model name if changed. My error is posted below. 

#current error when ran
Traceback (most recent call last):
  File "C:\Users\alex_\PycharmProjects\rsi\TDI-Trade.py", line 324, in <module>
    pricing_streaming()
  File "C:\Users\alex_\PycharmProjects\rsi\TDI-Trade.py", line 210, in pricing_streaming
    X = np.column_stack(
  File "<__array_function__ internals>", line 180, in column_stack
  File "C:\Users\alex_\PycharmProjects\rsi\venv\lib\site-packages\numpy\lib\shape_base.py", line 656, in column_stack
    return _nx.concatenate(arrays, 1)
  File "<__array_function__ internals>", line 180, in concatenate
ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 34 and the array at index 5 has size 68
