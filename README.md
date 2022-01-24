# TOO-THE-MOON-Trading-Bot-

Using the NASDAQ 100 as an indicator, this trading algorithm will buy/sell into TQQQ and sell/buy into SQQQ. Once fully optimized your dream of going to the MOON might possibly be achieved but be warned all trading comes with risk. Use this model at you own risk and be warned that the models used for this algorithm have not been fully optimized.

---
## Technologies

This project uses python 3.7 along with jupyter lab 3.0.14 with the following packages:


* [pandas](https://github.com/pandas-dev/pandas) - For data manipulation and analysis

* [hvplot](https://hvplot.holoviz.org/) - For data visualization only in testing

* [numpy](https://numpy.org/) - To create arrays for LSTM model

* [yfinance](https://pypi.org/project/yfinance/) - To get historical and current data from yahoo finance

* [finta](https://github.com/peerchemist/finta) - To create trading signals used in predictions

* [sklearn](https://scikit-learn.org/stable/) - To scale and analyse datasets

* [tensorflow](https://www.tensorflow.org/) - Library used to develop and train ML models

* [keras](https://keras.io/) - Creating deep learning models (included in tensorflow library)

* [alpaca](https://pypi.org/project/alpaca-trade-api/) - To buy and sell stocks

---
## Installation Guide

Before running the application first install the following dependencies:

```python
$ pip install pandas
$ pip install -U scikit-learn
$ conda install -c pyviz hvplot
$ conda install numpy
$ pip install --upgrade tensorflow
$ pip install yfinance
$ pip install finta
$ pip install alpaca-trade-api

```

---
## Usage

This is a trading function used to get the current data for the NASDAQ 100 (^NDX) and predict using three models (LSTM, Dense, and MLP Classifier) to decide whether or not to buy into TQQQ, which moves together with NASDAQ 100, and sell on SQQQ, which moves inverse to NASDAQ 100, or vice versa if a sell signal appears as the majority among the models. The initial run requires the user to input either a 1, -1, or 0. For a 1 the model will only act when it gets a sell signal from NASDAQ 100. For -1 it is the opposite and will only act with a buy signal. For 0 it will perform the first action that occurs. Please not the after running the function once reassign the variable in then buy and sell function with the output from the first iteration.

The models that are used have a accuracy for predicting buy and sell signals ranging from 50-55%. Due to limited time the models have not been fully optimized.

---
## Contributors
This code was created in 2022 for a project at (Education Services at UCB). 

Israel Fernandez -- 

Deep Patel -- Deep4Patel9@gmail.com

Dominik Tortes -- 

---
## License
MIT License

Copyright (c) 2021  

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
