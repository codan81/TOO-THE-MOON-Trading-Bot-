

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


def lstm_pred_to_signal(lstm_pred, X, scaler):
  pred = scaler.inverse_transform(lstm_pred)

  #create Visualization for Adj Close Prediction
  
  pred_df = pd.DataFrame({'Predicted':pd.concat([X.iloc[[-1],], pred], axis=0)})
  pred_df['Pred_RTN'] = pred_df['Predicted'].pct_change()
  pred_df = pred_df.dropna()
  pred_df['LSTM_Signal'] = 0.0

  # Generate Signal to buy stock[1] in Buy column
  pred_df.loc[(pred_df['Pred_RTN'] > 0), 'LSTM_Signal'] = 1

  # Generate Signal to sell stock[1] in Sell column
  pred_df.loc[(pred_df['Pred_RTN'] < 0), 'LSTM_Signal'] = -1

  return pred_df['LSTM_Signal'].values


def get_signal(pred):
    




#Load in NASDAQ 100 data
df = yf.download(tickers ='^NDX', period='4m' , interval='1d')
X_lstm = df['Adj Close'][-64:]
trading_df, in_list = get_indicators(df)
X = trading_df[in_list][-1]


# Load in Models
lstm = tf.keras.load_model('')
dense = tf.keras.load_model('') 
mlp = tf.keras.load_model('') 

p_lstm = lstm.predict(X_lstm)
p_dense = dense.predict(X)
p_mlp = mlp.predict(X)



new_signal = sum([
        lstm_pred_to_signal(p_lstm),
        get_signal(p_dense),
        get_signal(p_mlp)
        ])

if new_signal+signal == 0:
    
    if new_signal == 1:
        #Buy with TQQQ. Sell with SQQQ

    elif new_signal == -1:
        #Sell with TQQQ. Buy with SQQQ

    return new_signal

else:

    return signal