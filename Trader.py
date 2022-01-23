
def get_lstm_scaled_data(tickers, period, interval):
  original = yf.download(tickers =tickers, period=period , interval=interval)
  x = original['Adj Close'][-64:]
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaler.fit(original)
  x_scaled = scaler.transform(x)
  return scaler, x_scaled


def get_indicators(df):
  # Use Indicators from TA to get trading indicators for stock
  bbands_df = TA.BBANDS(df)
  mfi_df = pd.DataFrame({'MFI':TA.MFI(df)})
  macd_df = TA.MACD(df)
  smas_df = pd.DataFrame({'SMA_Short': TA.SMA(df, period=20)})
  smal_df = pd.DataFrame({'SMA_Long': TA.SMA(df, period=120)})
  indicators_list = [bbands_df.columns, mfi_df.columns, macd_df.columns, smas_df.columns, smal_df.columns]
  trading_signals_df = pd.concat([df, bbands_df, mfi_df, macd_df, smas_df, smal_df], axis=1)
  return trading_signals_df, indicators_list


def get_signals_scaled_data(tickers, period, interval):
  original = yf.download(tickers ='^NDX', period='MAX' , interval='1d')
  df, in_list = get_indicators(original)
  df = df[in_list]
  x = df[in_list][-1:]
  scaler = StandardScaler()
  scaler.fit(df)
  x_scaled = scaler.transform(x)
  return scaler, x_scaled


def lstm_pred_to_signal(lstm_pred, X, scaler):
  pred = scaler.inverse_transform(lstm_pred)
  pred_df = pd.DataFrame({'Predicted':pd.concat([X.iloc[[-1],], pred], axis=0)})
  pred_df['Pred_RTN'] = pred_df['Predicted'].pct_change()
  pred_df = pred_df.dropna()
  pred = []
  # Generate Signal to buy stock[1] in Buy column
  if pred_df['Pred_RTN'] >= 0:
    pred.append(1)
  # Generate Signal to sell stock[1] in Sell column
  elif pred_df['Pred_RTN'] < 0:
    pred.append(-1)
  return pred


def get_signal(pred):
  signal = []
  for i in range(0, len(pred)):
    if pred[i][0] >= pred[i][1]:
      signal.append(1)
    elif pred[i][0] < pred[i][1]:
      signal.append(-1)
  return signal



def buy_and _sell(signal)
  #Load in X_lstm and X data
  tickers = '^NDX'
  period = 'MAX'
  interval = '1d'
  mm_scaler, X_lstm = get_lstm_scaled_data(tickers, period, interval)
  std_scaler, X = get_signals_scaled_data(tickers, period, interval)

  # Load in Models
  lstm = tf.keras.load_model('')
  dense = tf.keras.load_model('') 
  mlp = tf.keras.load_model('') 

  p_lstm = lstm.predict(X_lstm)
  p_dense = dense.predict(X)
  p_mlp = mlp.predict(X)

  new_signal = sum(
          lstm_pred_to_signal(p_lstm, X_lstm, mm_scaler)+
          get_signal(p_dense)+
          get_signal(p_mlp)
          )

  if (new_signal % 2) != 0:
    if ((signal > 0  and new_signal > 0) or (signal < 0  and new_signal < 0)):
      signal = signal
    else:
      if new_signal == 1:
        Print("Buy with TQQQ. Sell with SQQQ")
        signal = 1
      
      elif new_signal == -1:
        Print("Sell with TQQQ. Buy with SQQQ")
        signal = -1
  else:
    signal = signal
      
  return signal