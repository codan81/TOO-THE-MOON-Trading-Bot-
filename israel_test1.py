#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[159]:


# Import Libraries and dependencies
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
import yfinance as yf
from finta import TA
import hvplot.pandas
import holoviews as hv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import keras_tuner as kt
from tensorflow import keras


# In[2]:


#Get Nasdaq 100(^NDX) for 1d from yahoo finance for MAX period.
nasdaq100_df = yf.download(tickers='^NDX', period='1y', interval='1h')
display(nasdaq100_df)


# In[3]:


# Visualize Close data for nasdaq100_df
close_plt = nasdaq100_df['Close'].hvplot(kind='line', title='Nasdaq 100 Close Prices', xlabel='Date', ylabel='Price ($)', line_color='grey')
close_plt


# In[4]:


# Set period to be used for trading indicators
period = 20


# In[29]:


# Use BBANDS, MFI and OBV from TA to get trading indicators for nasdaq100_df
bbands_df = TA.BBANDS(nasdaq100_df, period=period)
mfi_df = pd.DataFrame({'MFI':TA.MFI(nasdaq100_df, period=period)})
obv_df = pd.DataFrame({'OBV':TA.OBV(nasdaq100_df)})

display(bbands_df)
display(mfi_df)


# In[40]:


obv_df["OBV_EMA"]=obv_df.ewm(com=20).mean()
pd.options.display.float_format = '{:.6f}'.format
obv_df


# In[41]:


#Visualize data for bbands_df, mfi_df and obv_df
bbands_plt = bbands_df.hvplot(kind="line", xlabel='Datetime', ylabel='Price ($)', title='Bolinger Bands for NASDAQ 100')
mfi_plt = mfi_df.hvplot(kind='line', xlabel='Datetime', ylabel='MFI Values', title='Money Flow Index for NASDAQ 100')
obv_plt = obv_df.hvplot(kind='line', xlabel='Datetime', ylabel='OBV Values', title='On Balance Volume for NASDAQ 100')

mfi_sell_line = hv.HLine(80).opts(color='red', line_dash='dashed', line_width=2.0)
mfi_buy_line = hv.HLine(20).opts(color='green', line_dash='dashed', line_width=2.0)
display(bbands_plt*close_plt)
display(mfi_plt*mfi_buy_line*mfi_sell_line)
display(obv_plt)


# In[42]:


# Create a copy of nasdaq100_df and concatenate with bbands_df and mfi_df
trading_signals_df = nasdaq100_df.copy()
trading_signals_df = pd.concat([trading_signals_df, bbands_df, mfi_df, obv_df], axis=1)
display(trading_signals_df)


# In[43]:


# Create a trading algorithm using Bollinger Bands
# Set the Signal column
trading_signals_df["Signal_BB"] = 0.0

# Create a value to hold the initial trade signal
trade_signal = 0

# Generate the trading signals 1 (entry) or -1 (exit) for a long position trading algorithm
# where 1 is when the Close price is less than the BB_LOWER window
# where -1 is when the Close price is greater the the BB_UPPER window
# trading signal adds one for each buy and doesnt buy more until a sell which resets the trade signal back to 0 and vice verca
for index, row in trading_signals_df.iterrows():
    if (row["Close"] < row["BB_LOWER"]) and (trade_signal < 1):
        trading_signals_df.loc[index, "Signal_BB"] = 1.0
        trade_signal += 1
    elif (row["Close"] > row["BB_UPPER"]) and (trade_signal > 0):
        trading_signals_df.loc[index,"Signal_BB"] = -1.0
        trade_signal = 0

trading_signals_df['Signal_BB'].value_counts()


# In[44]:


# Create a trading algorithm using Money Flow Index
# Set Signal column
trading_signals_df['Signal_MFI'] = 0.0

# Create a value to hold the initial trade signal
trade_signal = 0

# Generate the trading signals 1 (entry) or -1 (exit) for a long position trading algorithm
# where 1 is when the MFI is less than the 20 (Oversold)
# where -1 is when the MFI is greater than 80 (Overbought)
# trading signal adds one for each buy and doesnt buy more until a sell which resets the trade signal back to 0 and vice versa
for index, row in trading_signals_df.iterrows():
    if (row['MFI'] > 80) and (trade_signal > 0):
        trading_signals_df.loc[index, 'Signal_MFI'] = -1
        trade_signal = 0
    elif (row['MFI'] < 20) and (trade_signal < 1):
        trading_signals_df.loc[index, 'Signal_MFI'] = 1
        trade_signal += 1
        
trading_signals_df['Signal_MFI'].value_counts()


# In[45]:


# Create a trading algorithm using On Balance Volume
# Set Signal column
trading_signals_df['Signal_OBV'] = 0.0

# Create a value to hold the initial trade signal
trade_signal = 0

# Generate the trading signals 1 (entry) or -1 (exit) for a long position trading algorithm
# where 1 is when the MFI is less than the 20 (Oversold)
# where -1 is when the MFI is greater than 80 (Overbought)
# trading signal adds one for each buy and doesnt buy more until a sell which resets the trade signal back to 0 and vice versa
for index, row in trading_signals_df.iterrows():
    if (row['OBV'] > row['OBV_EMA']) and (trade_signal > 0):
        trading_signals_df.loc[index, 'Signal_OBV'] = -1
        trade_signal = 0
    elif (row['OBV'] < row["OBV_EMA"]) and (trade_signal < 1):
        trading_signals_df.loc[index, 'Signal_OBV'] = 1
        trade_signal += 1
        
trading_signals_df['Signal_OBV'].value_counts()


# In[46]:


def signal_plot(df, signal):
    # Visualize entry position relative to close price
    entry = df[df[signal] == 1.0]["Close"].hvplot.scatter(
        color='green',
        marker='^',
        size=200,
        legend=False,
        ylabel='Price ($)',
        width=1000,
        height=400
    )

    # Visualize exit position relative to close price
    exit = df[df[signal] == -1.0]["Close"].hvplot.scatter(
        color='red',
        marker='v',
        size=200,
        legend=False,
        ylabel='Price ($)',
        width=1000,
        height=400
    )
    
    return entry*exit

display(signal_plot(trading_signals_df, 'Signal_BB')*close_plt.opts(title='Bolinger Bands Trading Strategy')*bbands_plt)

display(signal_plot(trading_signals_df, 'Signal_MFI')*close_plt.opts(title='Money Flow Index Trading Strategy'))

display(signal_plot(trading_signals_df, 'Signal_OBV')*close_plt.opts(title='On Trading Strategy'))


# In[47]:


# Set trading signals by using daily returns
trading_signals_df['Actual_Returns'] = trading_signals_df['Adj Close'].pct_change()
trading_signals_df = trading_signals_df.dropna()
# Initialize the new Signal column
trading_signals_df['Signal_RTN'] = 0.0

# Generate Signal to buy stock long
trading_signals_df.loc[(trading_signals_df['Actual_Returns'] >= 0), 'Signal_RTN'] = 1

# Generate Signal to sell stock short
trading_signals_df.loc[(trading_signals_df['Actual_Returns'] < 0), 'Signal_RTN'] = -1

trading_signals_df['Signal_RTN'].value_counts()


# In[48]:


# Find the Strategy Returns for the trading strategy. and visualize comparision of actual vs strategy
def compare_returns(df, signal):
    df['Strategy_Returns'] = df['Actual_Returns'] * df[signal].shift()
    return (1 + df[['Actual_Returns','Strategy_Returns']]).cumprod().hvplot(title=f'{signal} Strategy Returns Vs Actual Returns')

compare_returns(trading_signals_df, 'Signal_RTN')


# In[50]:


# Set X and y input for NN
X = trading_signals_df[['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'MFI', 'OBV']].shift().dropna().copy()
y = trading_signals_df['Signal_RTN']

display(X.head())
display(y[:5])


# In[51]:


# Set training start and end dates using DateOffset
# Set offset in years to offset data
offset = 6 
training_begin = X.index.min()
training_end = X.index.min() + DateOffset(months= offset)
print(f'Training Start: {training_begin}, Training End: {training_end}')


# In[52]:


# Set X_train, y_train, X_test, y_test
x_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]
x_test = X.loc[training_end:]
y_test = y.loc[training_end:]


# In[53]:


# Scale X_training and X_testing sets using StandardScaler()/MinMaxScaler()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled


# In[54]:


# Create data for using the last 60 signals worth of trading indicators to predict the signal for the next instance
X_train_ar = []
y_train_ar = []

for i in range(60, len(x_train_scaled)):
    X_train_ar.append(x_train_scaled[i-60:i,0])
    y_train_ar.append(y_train[i,0])


# In[55]:


# Create data array for testing data
X_test_ar = []
y_test_ar = y_test[60:]

for i in range(60, len(x_test_scaled)):
    X_test_ar.append(x_test_scaled[i-60:i])


# In[145]:


# Convert data to numpy arrays and then check shape for LSTM(3 dimensional)
X_train_ar, y_train_ar = np.array(X_train_ar), np.array(y_train_ar)
X_train_ar.shape


# In[144]:


X_test_ar= np.array(X_test_ar)
X_test_ar.shape


# In[150]:


# # Set up the Nueral Network model
# nn = Sequential()
# nn.add(Dense(50, activation="Relu",return_sequences=True, input_shape=(X_train_ar.shape[1], X_train_ar.shape[1])))
# nn.add(LSTM(50, activation="Rlu", return_sequences=False))
# nn.add(Dense(25, activation=None))
# nn.add(Dense(1, activation=None))


# In[160]:


def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model


# In[161]:


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5)


# In[170]:


tuner.search(X_train_ar, y_train, epochs=5, validation_data=(X_train_ar, y_train_ar))
best_model = tuner.get_best_models()[0]


# In[152]:


# Compile the Nueral Network model
nn.compile(optimizer='adam', loss='binarry_crossentropy')


# In[153]:


# Fit the model with training data
nn.fit(x_train_scaled, y_train, epochs=10)


# In[133]:


predictions = nn.predict(x_test_scaled)
predictions[:5]


# In[104]:


pd.options.display.float_format = '{:.2f}'.format
pred_df = pd.DataFrame({'Actual':y_test, 'Prediction':np.ravel(predictions)})
pred_df


# In[105]:


# Create a new column with a value of 1 for buy or 0 for sell using prediction and rounding.
pred_df.loc[(pred_df['Prediction'] >= 0.5), 'Pred_R'] = 1
pred_df.loc[(pred_df['Prediction'] < 0.5), 'Pred_R'] = 0
pred_df['Pred_R'].value_counts()


# In[106]:


# Visualize Pred_R and Actual
pred_df[['Actual', 'Pred_R']].hvplot(title='Actual Vs Predicted Signals')


# In[92]:


# Concatenate Actual_Returns to pred_df
pred_df = pd.concat([pred_df, trading_signals_df['Actual_Returns']], axis=1).dropna()


# In[93]:


# Visualize Return using Predicted 
compare_returns(pred_df, 'Pred_R')


# In[94]:


print(classification_report(pred_df['Actual'] ,pred_df['Pred_R']))


# In[ ]:




