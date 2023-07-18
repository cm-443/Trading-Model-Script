# tdi
Model and trading script using the trader's dynamic index. 

This 





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
