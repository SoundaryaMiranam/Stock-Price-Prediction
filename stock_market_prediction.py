pip install yfinance

import yfinance as yf
from textblob import TextBlob
import tweepy 
from tweepy import OAuthHandler
import os
import sys
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# keys and tokens from the Twitter 
consumer_key = '*************************************'
consumer_secret = '***************************************************'
access_token = '**************************************************'
access_token_secret = '*****************************************'

# attempt authentication 
try: 
    # create OAuthHandler object 
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    # set access token and secret 
    auth.set_access_token(access_token, access_token_secret)
    # create tweepy API object to fetch tweets 
    api = tweepy.API(auth)
except: 
    print("Error: Authentication Failed")

query = input("Please enter a stock quote from the CSV to perform the search: ").upper()

def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
     list_of_tweets = api.search(quote, count=num_tweets)
     
     polarity = 0
     positive = 0
     negative = 0
     neutral = 0
     null = 0

     for tweet in list_of_tweets:
       analysis = TextBlob(tweet.text)
       #print(analysis)
       polarity = analysis.sentiment.polarity
       #print(polarity)
       if analysis.subjectivity == 0:  
          null += 1
          next
       if polarity < 0:
         negative += 1
       elif polarity == 0:
            neutral += 1
       else:
            positive += 1

     if positive > ((num_tweets - null)/2):
        return True

def get_historical(stock_quote):
  # Download data from yahoo finance
  if yf.Ticker(stock_quote) != 400:
    value = yf.Ticker(stock_quote)
    hist = value.history(period="5y")
    hist.to_csv('hist.csv')
  return True  


from sklearn.preprocessing import MinMaxScaler


def stock_prediction():
   # Collect data points from csv
   df=pd.read_csv('/content/hist.csv')
   df1=df.reset_index()['Close']
   scaler=MinMaxScaler(feature_range=(0,1))
   df1=scaler.fit_transform(np.array(df1).reshape(-1,1))   

   # convert an array of values into a dataset matrix
   def create_dataset(dataset, time_step=1):
        val = (len(dataset)-time_step-1)
       	dataX, dataY = [], []

       	for i in range(val):
	        	a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
	        	dataX.append(a)
	        	dataY.append(dataset[i + time_step, 0])
       	return np.array(dataX), np.array(dataY)

   # reshape into X=t,t+1,t+2,t+3 and Y=t+4
   time_step = 100
   trainX, trainY = create_dataset(df1, time_step)

   # reshape input to be [samples, time steps, features] which is required for LSTM
   trainX =trainX.reshape(trainX.shape[0],trainX.shape[1] , 1)   

   # Create and fit Multilinear Perceptron model
   model = Sequential()
   # dense layer is a layer of neurons where each neuron of this layer is connected to each neuron of next layer
   # add dense layer of output shape (*,8) and input having 1 column
   model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
   model.add(LSTM(50,return_sequences=True))
   model.add(LSTM(50))
   model.add(Dense(1))   
   model.compile(loss='mean_squared_error', optimizer='adam')
   # train in batches of 2 in 100 epochs 
   # trainX has current stock value of time t and trainY has future stock value of time t+1
   model.fit(trainX, trainY, epochs=100, batch_size=25, verbose=2)

   # Lets Do the prediction and check performance metrics
   train_predict=model.predict(trainX)

   # Transformback to original form
   train_predict=scaler.inverse_transform(train_predict)
   array_length = len(train_predict)
   val = train_predict[array_length - 1]


   # Our prediction for tomorrow
   result = print("The price will move from", (df['Close'].iloc[-1]),"to" , str(val)[1:-1])
   return result


# Check if the stock sentiment is positve
if not stock_sentiment(query, num_tweets=100):
    print('This stock has bad sentiment, please re-run the script')
    sys.exit()


# Check if we got the historical data
if not get_historical(query):
    print('Google returned a 404, please re-run the script and')
    print('enter a valid stock quote from NASDAQ')
    sys.exit() 


# We have our file so we create the neural net and get the prediction
print(stock_prediction())
