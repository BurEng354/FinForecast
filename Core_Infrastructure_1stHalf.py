##################

##Script for the AI Financial Forecaster's Core Infrastructure's 1st half from Team 1(CE903)

##################

#!pip install nasdaq-data-link
import os
import nasdaqdatalink ##Readme: Install this
import pandas_datareader.data as web ##Readme: Install this


import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer, KNNImputer
#from sklearn.linear_model import LinearRegression

######################################
######################################
##Creating Project Directory for Stock Market Data Storage (use minimally due to Space Constraints)
Leaf_Directory = "Stock Market Datasets"
Parent_Directory = 'C:\Group Project'

#Storage Path is Global
storage_path = os.path.join(Parent_Directory, Leaf_Directory)

try:
    os.makedirs(storage_path)
    print("Directory '%s' created successfully" % Leaf_Directory)
except OSError as error:
    print("Directory '%s' already created" % Leaf_Directory)

#Note: error was defined for Error Handling

#######################################
#######################################
def historical_data_recorder(source):
    '''
    Function Algorithm:
        
    1) Depending on the Source chosen by the User:
        Configure the free Nasdaq Data Link API or the Pandas Datareader for Yahoo Finance
    
    2) In either case: Obtain Stock Market Dataset from the chosen Source
        based on User Input at the Python Terminal. 
    
    3) Save said Dataset at a certain Project Directory/Device Storage (overwrite if needed).
    
    4) And finally, return the Dataframe of the Stock_Market_Dataset along with some other Information
    
    Recommendations:
        
    1) Error Handling for cases when overwriting has to be performed on an alredy 
        open .csv file
    '''
    
    if source == "Nasdaq API":
        #Steps 2, 3 and 4 for Nasdaq API
        ##Configuring the Financial Data Providers for Historical Data Capturing (APIs and more...)
        api_key = "vZTqgD6CcX4SeDsExcCG" ##Key obtained by Burhan Awan of Team 1 using Nasdaq's Website
        nasdaqdatalink.ApiConfig.api_key = api_key
        NASDAQ_DATA_LINK_API_KEY = api_key#GLobal
    
        ##Histoircal (Structured) Stock Market Data Capture
        ##Obtain User Input    
        database = input("Provide the DataBase Name from Nasdaq Data Link API (e.g. WIKI): ")
        company_ticker = input("\nProvide the Company Ticker from Nasdaq Data Link API (e.g. AAPL/MSFT): ")
        DataLink_Code = database + '/' + company_ticker 
    
        print("\nProvide Temporal Data for the Stock Market Dataset: ")
        start = input("\nPlease mention the Start Date (YYYY-MM-DD): ")
        end = input("\nPlease mention the End Date (YYYY-MM-DD): ")
        interval_timesteps = input("\nPlease mention the Interval between Timesteps (daily/monthly/annual). Note: Only chose daily for now : ")#Frequency
    
        ##Recommnedation from Burhan Awan: Use Detailed Guide of Nasdaq Data Link for Data Capture
        
        ##Derived from Burhan Awan's Code
        Stock_Market_Dataset = nasdaqdatalink.get(DataLink_Code, start_date = start, end_date = end, collapse = interval_timesteps)
        Stock_Market_Dataset.to_csv("C:\Group Project\Stock Market Datasets\ %s.csv" % company_ticker)
        print ("Stock Market Dataset named %s.csv  is saved" % company_ticker)
        return company_ticker, start, end, Stock_Market_Dataset

    if source == "Yahoo Finance":
        #Steps 2, 3 and 4 for Yahoo Finance
        company_ticker = input("\nProvide the Company Ticker from Yahoo Finance (e.g. GE/GOOG): ")
        print("\nProvide Temporal Data for the Stock Market Dataset: ")
        start = input("\nPlease mention the Start Date (YYYY-MM-DD): ")
        end = input("\nPlease mention the End Date (YYYY-MM-DD): ")         
        
        ##Dervied from Sundar Acharya's Code
        Stock_Market_Dataset = web.DataReader(str(company_ticker), 'yahoo', start, end, api_key=None)
        
        Stock_Market_Dataset.to_csv("C:\Group Project\Stock Market Datasets\ %s.csv" % company_ticker)
        print ("Stock Market Dataset named %s.csv  is saved" % company_ticker)
        return company_ticker, start, end, Stock_Market_Dataset

####################################
####################################
def stock_market_dataset_preprocessor(dataset):
    '''
    Funtion Algorithm:
    
    1) Put in NaN Values to make the Time Series regular at frequency "daily" (
        for processing by LSTM or other RNNs)
        https://www.ibm.com/docs/en/streams/5.3?topic=series-regular-irregular-time
    
    
    2) For Technical Reasons like LSTM Processing: The Stock Market Data 
        corresponding to Closing Days of Exchanges will be derived using Assumptions
        and Linear Interpolation
        
    Therefore: AI Forecaster presupposes what Stock Market Data will be on 
        Closing Days for Tehnical Reasons
    '''
    #Step 1
    dataset = dataset.asfreq(freq ='1D')
    #Note: Index of dataset has datetime objects

    ##Dropping Volume and Adj. Volume might be attempted to reduce Space
    ##dataset = dataset.drop(['Volume', 'Adj. Volume'], axis = 1)
    
    #Step 2
    imputed_dataset = pd.DataFrame()
    for column in dataset.columns:
        imputed_dataset[column] = dataset[column]
        imputed_dataset[column].interpolate(limit_direction = "both", inplace = True)

    '''
    Recommendations
    Using MICE Imputer is not recommended (see line plot for MICE Imputer in Report). 
    Also, KNN Imputer gave similar results
    
    Code used:
    mice_imputer = IterativeImputer(estimator = LinearRegression(), missing_values = np.nan, max_iter = 30, imputation_order='roman')
    imputed_dataset = mice_imputer.fit_transform(dataset)#also removes Date column and gives a Numpy ndarray
    imputed_dataset = pd.DataFrame(imputed_dataset, columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio',
       'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'])   
    '''
    return imputed_dataset #, dataset
