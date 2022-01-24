import pandas as pd
import numpy as np
import yfinance as yf
from finta import TA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pickle


# Load .env environment variables
load_dotenv()

# Set api keys in .env file equal to these strings for key and secret to trade
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"
api = tradeapi.REST(API_KEY, API_SECRET, ALPACA_API_BASE_URL, api_version="v2")



# Function to get and scale lstm data and returns that last 64 as a matrix to predict on.
def get_lstm_scaled_data(tickers, period, interval):
  original = yf.download(tickers =tickers, period=period , interval=interval)
  x_train = pd.DataFrame(original['Adj Close'][:-64])
  x = pd.DataFrame(original['Adj Close'][-64:])
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaler.fit(x_train)
  x_scaled = scaler.transform(x)
  x_array = np.reshape(x_scaled, (1,64,1))
  return scaler, x_array



# Function to get all indicators to predict with and returns a list of indicators and new df with indicators
def get_indicators(df):
  # Use Indicators from TA to get trading indicators for stock
  bbands_df = TA.BBANDS(df)
  mfi_df = pd.DataFrame({'MFI':TA.MFI(df)})
  macd_df = TA.MACD(df)
  obv_df = pd.DataFrame({'OBV':TA.OBV(df)})
  in_list_d = bbands_df.columns.values.tolist() + mfi_df.columns.values.tolist() + macd_df.columns.values.tolist()
  in_list_mlp = bbands_df.columns.values.tolist() + mfi_df.columns.values.tolist() + ['OBV']
  trading_signals_df = pd.concat([df, bbands_df, mfi_df, macd_df, obv_df], axis=1)
  return trading_signals_df, in_list_d, in_list_mlp



# Function to get signals and scale them with standardscaler and outputs the X values to predict with with Dense and MLP.
def get_signals_scaled_data(tickers, period, interval):
  original = yf.download(tickers =tickers, period=period , interval=interval)
  df, in_list_d, in_list_mlp = get_indicators(original)
  d_df = df[in_list_d][:-1]
  x_d = df[in_list_d][-1:]
  mlp_df = df[in_list_mlp][:-1]
  x_mlp = df[in_list_mlp][-1:]
  scaler = StandardScaler()
  scaler.fit(d_df)
  xd_scaled = scaler.transform(x_d)
  scaler_mm = MinMaxScaler(feature_range=(0,1))
  scaler_mm.fit(mlp_df)
  xmlp_scaled = scaler_mm.transform(x_mlp)
  return scaler_mm, xd_scaled, xmlp_scaled



# Function to change lstm value to adj close prediction and then calculate pct_change to get predicted RTN for next interval
def lstm_pred_to_signal(lstm_pred, scaler, tickers, interval):
  unscaled = scaler.inverse_transform(lstm_pred)
  X = yf.download(tickers = tickers, period='5D' , interval=interval)
  pred_val = (unscaled.sum() - X['Adj Close'][-1:])/unscaled.sum() 
  pred = []
  # Generate Signal to buy stock[1] in Buy column
  if pred_val[-1] >= 0:
    pred.append(1)
  # Generate Signal to sell stock[1] in Sell column
  elif pred_val[-1] < 0:
    pred.append(-1)
  return pred



# Function to get_signals for Buy Sell column that outputs from Dense and MLP predicion (whichever value is greater)
def get_signal(pred):
  signal = []
  if pred[0][0] >= pred[0][1]:
    signal.append(1)
  elif pred[0][0] < pred[0][1]:
    signal.append(-1)
  return signal


def get_mlp_signal(pred):
  signal=[]
  if pred == 0:
    signal.append(-1)
  elif pred == 1:
    signal.append(1)
  return signal

# Function to buy stocks on alpaca
def alpaca_buy(ticker, stocks_to_buy):
  prices = api.get_barset(ticker, "1Min").df
  limit_amount = prices[ticker]["close"][-1]
  # Submit order
  api.submit_order(
      symbol=ticker, 
      qty=stocks_to_buy, 
      side='buy', 
      time_in_force="gtc", 
      type="limit", 
      limit_price=limit_amount
  )



# Function to sell stocks on alpaca (***Has to check if stocks are held before selling)
def alpaca_sell(ticker, stocks_to_sell):
  prices = api.get_barset(ticker, "1Min").df
  limit_amount = prices[ticker]["close"][-1]
  # Submit order
  api.submit_order(
      symbol=ticker, 
      qty=stocks_to_sell, 
      side='sell', 
      time_in_force="gtc", 
      type="limit", 
      limit_price=limit_amount
  )



# Please set buy_and_sell equal to signal and set after first iteration set input variable signal.
# For first iteration please choose either -1, 1, 0:
#           -1 it will buy next for TQQQ and sell for SQQQ if new_signal is >0
#            1 it will sell next for TQQQ and buy for SQQQ if new_signal is <0
#            0 will cause whatever new_signal that appears after first iteration to run for buy and sell
def buy_and_sell(signal):
  #Load in X_lstm and X data
  tickers = '^NDX'
  period = 'MAX'
  interval = '1d'
  lstm_scaler, X_lstm = get_lstm_scaled_data(tickers, period, interval)
  mlp_scaler, X_d, X_mlp = get_signals_scaled_data(tickers, period, interval)
  
  # Please insert how many stocks you would like to buy here.
  stocks_tqqq = 1
  stocks_sqqq = 1


  # Load in Models
  lstm = tf.keras.models.load_model('Proto_Models/LSTM_Proto_1855.545406705499.h5')
  dense = tf.keras.models.load_model('Proto_Models/Dense_Proto_0.5345016429353778%.h5')
  with open('Proto_Models/MLP_Proto.pkl', 'rb') as f:
    mlp = pickle.load(f)

  p_lstm = lstm.predict(X_lstm)
  p_dense = dense.predict(X_d)
  p_mlp = mlp.predict(X_mlp)

  new_signal = sum(
          lstm_pred_to_signal(p_lstm, lstm_scaler, tickers, interval)+
          get_signal(p_dense)+
          get_mlp_signal(p_mlp)
          )

  if ((new_signal % 2) != 0) or (new_signal == 2) or (new_signal == -2):
    if ((signal > 0  and new_signal > 0) or (signal < 0  and new_signal < 0)):
      signal = signal
      print(f'Predicted Signal same as previous signal: {new_signal}.')
    else:
      if new_signal == 1:
        # Input Alpaca buy/sell info
        alpaca_buy('TQQQ', stocks_tqqq)
        alpaca_sell('SQQQ', stocks_sqqq)
        print(f"Bought {stocks_tqqq} of TQQQ. Sold {stocks_sqqq} of SQQQ")
        signal = 1
      
      elif new_signal == -1:
        #input alpaca buy sell info
        alpaca_sell('TQQQ', stocks_tqqq)
        alpaca_buy('SQQQ', stocks_sqqq)
        signal = -1
        print(f"Sold {stocks_tqqq} of TQQQ. Bought {stocks_sqqq} of SQQQ")
        
  else:
    print(f'There was a tie. One predicted signals was 0. The sum of the 3 signals is {new_signal}.')
    signal = signal
      
  return signal
