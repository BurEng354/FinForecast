##################

##Script for the AI Financial Forecaster's Main Code and Trading Strategy Component from Team 1(CE903)

##################

#!pip install nasdaq-data-link
import Core_Infrastructure_1stHalf ##Also forms the Project Directory
from Core_Infrastructure_1stHalf import historical_data_recorder
from Core_Infrastructure_1stHalf import stock_market_dataset_preprocessor

import Core_Infrastructure_2ndHalf
from Core_Infrastructure_2ndHalf import getROC
from Core_Infrastructure_2ndHalf import willR
from Core_Infrastructure_2ndHalf import midPrice
from Core_Infrastructure_2ndHalf import TAA_Dataset_Transformer
from Core_Infrastructure_2ndHalf import TAI_Dataset_Preprocessor

import Trading_Strategy_1stHalf
from Trading_Strategy_1stHalf import MLP_model 

import Trading_Strategy_2ndHalf
from Trading_Strategy_2ndHalf import modelLSTM


import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


'''
###############################################
###############################################
MAIN CODE
'''

if __name__ == '__main__':
    while 1:
        '''
        Implementation of Stock Market Data Storage
        '''
        dataset_source = input("Choose source of Stock Market Dataset (Device Storage/ Nasdaq API/ Yahoo Finance): ")##Needs Error Handling due to User Input
        
        if dataset_source == "Device Storage":
            company_ticker = input("\nProvide the Company Ticker (Use Project Directory if needed): ")
            Stock_Market_Dataset = pd.read_csv("C:\Group Project\Stock Market Datasets\ %s.csv" % company_ticker)
            start = Stock_Market_Dataset['Date'].iloc[0]
            end = Stock_Market_Dataset['Date'].iloc[-1]
            Stock_Market_Dataset['Date'] = pd.to_datetime(Stock_Market_Dataset['Date'])
            Stock_Market_Dataset = Stock_Market_Dataset.set_index('Date')
            print("\nStock Market Dataset successfully accessed from Device Storage.\n" )
            #Note: The Index/Date Column now contains "datetime Objects (datetime64[ns])"
            
        if dataset_source == "Nasdaq API" or dataset_source == "Yahoo Finance":
            company_ticker, start, end, Stock_Market_Dataset = historical_data_recorder(dataset_source)
            #Note: The Index/Date Column contains "datetime Objects (datetime64[ns])"
            
        
        '''
        Implementations of Stock Market Data Preprocessor and TAA Dataset Transformer
        '''
        S_M_Dataset_copy = Stock_Market_Dataset
        S_M_Dataset_preprocessed = stock_market_dataset_preprocessor(S_M_Dataset_copy)
        Price_History, Prices_Dataframe, TA_Indicators_Dataset = TAA_Dataset_Transformer(S_M_Dataset_preprocessed, 'Close')        
        
        '''
        Implementaions for Line Plots for Time Series of: 
            Historical Close Price and Technical Analysis Indicators
        '''
        
        plot_cmd = input("Are Plots of CLose Price and Technical Analysis Indicators Required (Yes/No): ")
        if plot_cmd == "Yes":
            #Hist_fig = plt.figure()
            #Prices_Dataframe.plot(title = ("Close Price VS Timesteps for Stocks of %s" % company_ticker))
            #TA_Indicators_Dataset.set_index('Date')
            TA_Indicators_Dataset.set_index('Date').plot(subplots = True, layout=(4, 2), figsize = (12, 12), title = ("Technical Analysis Indicators for Stocks of %s" % company_ticker))
        
        
        '''
        Implementation of the Preprocessor of the Transformed Dataset/TAI Dataset 
        '''
        print("\nUnivariate LSTM is the Baseline ML Model.\n")
        ml_model = input("\nName the ML Model (e.g. MLP/uni_LSTM): ")
        MOA_model = input("\nName the Mode of Operation of the ML Model (e.g. Train/Predict): ")                 
        TAIinput_train_scaled, TAIinput_test_scaled, TAIoutput_train_scaled, TAIoutput_test_scaled, ip_scalar, op_scalar = TAI_Dataset_Preprocessor(TA_Indicators_Dataset, ml_model)
        
        
        '''
        Implementation of the Trading Strategy Component 
        MUST TRAIN AND PREDICT IN THE SAME RUN AS MODELS ARE NOT SAVED OR LOADED
        MUST TRAIN AND PREDICT IN THE SAME RUN AS MODELS ARE NOT SAVED OR LOADED
        '''
        if ml_model == "MLP":
            MLP_model(MOA_model, TAIinput_train_scaled, TAIoutput_train_scaled, TAIinput_test_scaled, op_scalar, TA_Indicators_Dataset)
                    
        #if ml_model == "uni_LSTM":#Univariate LSTM is the Baseline
        #Use Code from Trading_Strategy_2ndHalf here            
            
        ##To END the Program:
        end_cmd = input("Do you wish to Finish Now (Yes/No): ")
        if end_cmd == "Yes":
            break


