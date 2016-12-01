import pandas as pd
import quandl


# Get the dataframe from quandl
df = quandl.get("WIKI/GOOGL", authtoken="o_dsrgif3_EyBgzHK1sf")

# Re-create dataframe to just be these cols
# Open - Start of day price
# Close - End of day price
# Low - Lowest price of day
# High - Highest price of day
# Volume - Amount trades occured
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Define a new col for High - low percent volatility
# (High - Low)
# ------------  * 100
#     Low
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# Define a new col for percent change from start of day to end of day
# (Close - Open)
# -------------- * 100
#      Open
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# Define a new dataframe
# Only the cols we care about
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


# Print the five oldest entries
print(df.head())