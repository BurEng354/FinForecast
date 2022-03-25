#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:41:44 2022

@author: sundaracharya


###########################
###########################

PLEASE EXAMINE modelLSTM
PLEASE EXAMINE modelLSTM
PLEASE EXAMINE modelLSTM
PLEASE EXAMINE modelLSTM
PLEASE EXAMINE modelLSTM
PLEASE EXAMINE modelLSTM

###########################
###########################
"""
import pandas as pd
import pandas_datareader as web
import datetime as dt
from datetime import timedelta
import numpy as np
import keras.backend as KB
import tensorflow as tf

import matplotlib.pyplot as plt

import math


from keras.models import Sequential
from sklearn.model_selection import TimeSeriesSplit,cross_val_score,GridSearchCV
from keras.layers import Dense,LSTM,SimpleRNN
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


import joblib

import sklearn.metrics as metrics

def preprocessing(dataframe,startdate):
	##Burhan's Comments: Used drop_duplicates(subset=['Date']) but later set aside
    # removing duplicate rows, the last appearance will be dropped
    dataframe = dataframe.drop_duplicates(subset=['Date'])
    dataframe = dataframe.set_index('Date')
    # dataframe.index = pd.DatetimeIndex(dataframe.index)
    

	##Burhan's Comments: Not needed
    # interpolating missing dates to the dataset as we are working on time series data, we need regularity in data
    daterange = pd.date_range(startdate,dt.datetime.now())
    dataframe = dataframe.reindex(daterange,fill_value=np.nan)
    unsampled= dataframe.resample('D')
    
    	##Burhan's Comments: Not considered (lack of time to understand the advantage of this)
    # spline interpolation with polynomial value 2
    # linear interpolation to be implemented in both direction 
    dataframe_cpy = dataframe.interpolate(method='linear',order=2, limit_direction='both')
    # print(dataframe_cpy.dtypes)
    # dataframe_cpy=dataframe
    return dataframe_cpy

##Burhan: Needed and incorporated
#get stock data from different finances
def stockData(stock,startDate,financeName):
    if financeName=='yahoo':
        start = startDate
        end = dt.datetime.now()
        # columns = ['High', 'Low','Open','Close','Volume','Adj_Close']
        # df_ = pd.DataFrame(columns=columns)
        stdt = web.DataReader(str(stock),financeName, start, end,api_key=None).reset_index()
        # df_ = df_.append(stdt).reset_index()
        # df_ = df_.rename(columns={df_.columns[0]:'Date'})
        stdt.columns = stdt.columns.str.replace(' ','')
        stdt = stdt.rename(columns={stdt.columns[0]:'Date'})
        return stdt
    else:
        print("Source not available")

##Burhan: Used all these Functions    
#Rate of change calculation and return 
def getROC(adjClose):
    result = ((adjClose-adjClose.shift(1))/adjClose.shift(1))*100
    return result

# William%R calculation function
def wILLR(adjClose,lookback):
    result = []
    for i in range(len(adjClose)):
        value = (np.max(adjClose[i-lb:i])-adjClose[i])/(np.max(adjClose[i-lb:i])-np.min(adjClose[i-lb:i]))
        result.append(value)
    return result

    

# Midprice calculation
def midPrice(adjClose,lookback):
    result =[]
    for i in range(len(adjClose)):
        value = np.max(adjClose[i-lb:i])-np.min(adjClose[i-lb:i])
        result.append(value)
    return result

##Not Needed
# Function to print performance metrics of our regression model for visulalizing errors
def miscRegressionHelper(y_true,y_pred):
    mse = round(metrics.mean_squared_error(y_true,y_pred),6)
    rmse = round(np.sqrt(mse),6)
    r2 = round(metrics.r2_score(y_true,y_pred))
    
    print('MSE= ',mse)
    print('RMSE= ',rmse)
    print('r2 = ',r2)




#univariate long short term memory model 

def modelLSTM(dataframe,trainpredict,timerange):
    #setting training percentage
    training_data_percentage = 0.8
    #scaler for normalization
    scaler = MinMaxScaler(feature_range = (0,1))
    #condition according to user preferences
    # for the first time, it will go when trainpredict is train
    if(trainpredict=='train'):
        df = dataframe.filter(['Close'])
        
        # Converting dataframe to array
        dataset =df.values
        
        # calculation of total number of rows in our dataset
        training_data_len = math.ceil(len(dataset)*training_data_percentage)
        
        
        # Scaling the data 
        scaled_data = scaler.fit_transform(dataset)
        #print(scaled_data)
        
        # create the training data set 
        # create the scaled train data set
        train_df = scaled_data[0:training_data_len,:]
        
        # split the data into x_train and y_train dataset
        x_train_data = []
        y_train_data = []
        
        for i in range(timerange,len(train_df)):
            x_train_data.append(train_df[i-timerange:i,0])
            y_train_data.append(train_df[i,0])
        
        # Convert the x_train and y_train to numpy arrays
        x_train_data = np.array(x_train_data)
        y_train_data=np.array(y_train_data)
        
        # Reshaping the data to create 3-d data
        x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1)) #length of row, number of time steps, 1= number of features
        
        
        # Build the model
        model = Sequential()
        # 50 neurons, return sequence = true because we're going to use another lstm layer (to get output 3-D sequence)
        # input shape as this is our first layer (time steps, 1)
        model.add(LSTM(50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
        # another layer of LSTM model 
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(20))
        model.add(Dense(1))
        
        # compile the model using adam activation function and loss is calculated in MSE
        model.compile(optimizer='adam',loss='mean_squared_error')
        
        
        # Train the model with batch size and epochs
        model.fit(x_train_data,y_train_data,batch_size=2,epochs=10)
        
        
        # create the testing data set
        # create a new array containing scaled values
        test_data = scaled_data[training_data_len-timerange:,:]
        # create the data sets x_test and y_test
        x_test_data =[]
        y_test_data = dataset[training_data_len:,:]
        for i in range(timerange,len(test_data)):
            x_test_data.append(test_data[i-timerange:i,0])
            
        # convert test data to numpy array
        x_test_data = np.array(x_test_data)
        
        # reshape the data
        x_test_data = np.reshape(x_test_data,(x_test_data.shape[0],x_test_data.shape[1],1))
        
        # get the models predicted price values
        predictions = model.predict(x_test_data)
        predictions = scaler.inverse_transform(predictions)
        
        # Get root mean squared error (RMSE) of the model
        # lower values of mSE indicates a better fit.
        
        rmse = np.sqrt(np.mean(predictions-y_test_data)**2)
        # print(rmse)
        
        # store the trained model such that, it doesnot have to train on every request
        filename = "lstm_model_trained.joblib"
        joblib.dump(model,filename)
        


    else:
        #loading the previously trained model 
        load_model = joblib.load('lstm_model_trained.joblib')
        df_original=dataframe.filter(['Close'])
        final_predicted_price=0   
        cont = 1
        while(cont==1):
            df = df_original.copy()
            # getting day in future for prediction
            predict_for = int(input('Enter the day for which you are want to make prediction : '))
            for i in range(predict_for):
                # get last 60 days closing price and convert to dataframe array
                lookback_days_data = df[-timerange:].values
                
                # scale the data [0,1]
                lookback_days_data_scaled = scaler.fit_transform(lookback_days_data)
                # lookback_days_data_scaled = scaled_data.transform(lookback_days_data)
                
                # create an empty list
                X_test_df = []
                
                # Append the past lookback time data
                X_test_df.append(lookback_days_data_scaled)
                
                # convert the X_test data set to numpy array
                X_test_df = np.array(X_test_df)
                
                # Reshape the data
                X_test_df = np.reshape(X_test_df,(X_test_df.shape[0],X_test_df.shape[1],1))
                
                # Get the predicted scaled price
                pred_price = load_model.predict(X_test_df)
                
                # undoing the scaling
                pred_price = scaler.inverse_transform(pred_price)
                final_predicted_price=pred_price
                #adding the final predicted price to the df array to fill the data space
                df.loc[max(df.index)+timedelta(days=1),'Close']=final_predicted_price
            # print(df)
            print("The predicted price after "+str(predict_for)+" days is :"+ str(final_predicted_price))
            
            
            cont = int(input("Do you want to predict more with this model? \n 1. Enter '1' to predict again \n 2. Enter '2' to change model or trading parameter \n: "))            
        
            
        
        
        
        
    
def modelRF(dataset,trainpredict): 
    scaler = MinMaxScaler(feature_range = (0,1))
    
    if(trainpredict=='train'):
        # Scaling the data 
        
        scaled_data = scaler.fit_transform(dataset)
        
        training_data_len=math.ceil(len(dataset)*0.8)
        train, test = dataset.iloc[0:training_data_len,:],dataset.iloc[training_data_len:,:]
        
        #training test data with features and label
        X_train,y_train = train.drop(['AdjClose'],axis=1),train['AdjClose']
        X_test,y_test = test.drop(['AdjClose'],axis=1),test['AdjClose']
        model = RandomForestRegressor()
        #grid search hyperparameters for model optimality
        parameter_search = {
            'n_estimators':[5,11,15], # test by building 5,11,15 decision trees
            'max_features':['auto','log2'],
            'max_depth':[i for i in range(8,14)] # set max depth of the decision tree
            }
        #using timeseriessplit to get continuous data as we can't be random
        tcross_val = TimeSeriesSplit(n_splits=10)
        #estimating the best features in dataset
        grid = GridSearchCV(estimator=model,cv=tcross_val,param_grid=parameter_search,scoring='neg_root_mean_squared_error')
        grid.fit(X_train,y_train)
        best_model = grid.best_estimator_
        best_score = grid.best_score_
        
        
        # predicting
        y_true = y_test.values
        y_pred = best_model.predict(X_test)
        miscRegressionHelper(y_true,y_pred)
        
        # store the trained model such that, it doesnot have to train on every request
        filename = "rf_model_trained.joblib"
        joblib.dump(model,filename)
        
        # important_features = best_model.feature_importances_
        # features = X_train.columns
        # index_feat = np.argsort(important_features)
        
        # plt.figure(figsize=(20,10))
        # plt.title('Feature Importances')
        # plt.barh(range(len(index_feat)), important_features[index_feat], color='red', align='center')
        # plt.yticks(range(len(index_feat)), [features[i] for i in index_feat])
        # plt.xlabel('Relative Importance')
        # plt.show()
        
        # # It is found that only high,open and low are good features for the prediction
        # #now creating random forest with only two features 
        # rf_new = RandomForestRegressor(n_estimators=30,random_state=0)
        # print(X_train)
        # imp_features = [X_train.index('High'),X_train.index('Low'),X_train.index('Open')]
        # train_new = X_train[:,imp_features]
        # test_new = X_test[:,imp_features]
        
        # #training the random forest
        # rf_new.fit(train_new,y_train)
        
        # #making prediction
        # predictions = rf_new.predict(test_new)
        
        # #error calculation
        # miscRegressionHelper(y_test, predictions)
    
    
    else:
        #loading the previously trained model 
        load_model = joblib.load('rf_model_trained.joblib')
        df_original=dataset.filter(['Close'])
        final_predicted_price=0   
        lookback_days_data=2
        timerange=10
        cont = 1
        while(cont==1):
            df = df_original.copy()
            # getting day in future for prediction
            predict_for = int(input('Enter the day for which you are want to make prediction : '))
            # get last 60 days closing price and convert to dataframe array
            lookback_days_data = df[-timerange:].values
            
            # scale the data [0,1]
            lookback_days_data_scaled = scaler.fit_transform(lookback_days_data)
            # lookback_days_data_scaled = scaled_data.transform(lookback_days_data)
            
            # create an empty list
            X_test_df = []
            
            # Append the past lookback time data
            X_test_df.append(lookback_days_data_scaled)
            
            # convert the X_test data set to numpy array
            X_test_df = np.array(X_test_df)
            
            # Reshape the data
            X_test_df = np.reshape(X_test_df,(X_test_df.shape[0],X_test_df.shape[1],1))
            
            # Get the predicted scaled price
            pred_price = load_model.predict(X_test_df)
            
            # undoing the scaling
            pred_price = scaler.inverse_transform(pred_price)
            final_predicted_price=pred_price
            #adding the final predicted price to the df array to fill the data space
            df.loc[max(df.index)+timedelta(days=1),'Close']=final_predicted_price
    
    
    
    
    
    
  
    

# if __name__ == '__main__':
#     sd = dt.datetime(2021,8,1)
#     while(True):
#         stockSymbol = str(input("Enter the stock name, e.g. APPL: "))
#         exch = int(input("Select the Exchange name: \n 1 for 'yahoo'\n 2 for 'nasdaq': \n"))
#         finance=''
#         if exch==1:
#             finance ='yahoo'
#         elif exch==2:
#             finance='nasdaq'
#         else:
#             print('finance not selected')
#         dayRange = int(input('Select the range of time in days: '))
#         stData = stockData(stockSymbol,sd,finance)
#         #stData = stockData('AAPL',sd,finance)
        
#         # calculate rate of change and store in new column 
#         dataROC = getROC(stData['Close'])
#         stData['ROC']=dataROC
#         # calculate williams%R and store in new column
#         #set the lookback period for william r function
#         lb = 5
#         dataWILLR = wILLR(stData['Close'],lb)
#         stData['WILLR']=dataWILLR

#         # calculation of midprice and create new column to insert
#         lb=5 # if required to change as per different functions, we can change
#         midpriceData = midPrice(stData['Close'],lb)
#         stData['MidPrice']=midpriceData
        
#         #preprocess the data
#         stData = preprocessing(stData,sd)
#         stData.index.name='Date'
        
        
#         test_size=int(0.3 *pd.DataFrame(stData).shape[0])
#         train_data = stData[:-test_size]
#         test_data = stData[-test_size:]
#         whichModel = int(input("Select Models to implement: \n 1 for LSTM \n 2 for RF: \n"))
#         if(whichModel==1):
#             # training the LSTM model
#             modelLSTM(stData,'train',dayRange)
#             modelLSTM(stData, '', dayRange)
#         elif(whichModel==2):
#             modelRF(stData,'train')
#             modelRF(stData,'')
            
#         else:
#             print("invalid input")
    

'''
Deep Learning: LSTM
'''
# model_LSTM = Sequential()
# model_LSTM.add(LSTM(20, input_shape = (TAI_Dataset_preprocessed['EMA'].count(), 5)))
# model_LSTM.add(Dense(1, activation='relu'))
# #model_MLP.summary()
# model_LSTM.compile(loss = 'mse', optimizer = 'rmsprop', metrics = ['mae'])

