##################

##Script for the AI Financial Forecaster's Core Infrastructure's 2nd half from Team 1(CE903)

##################

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

##########################
##########################
#Rate of Change (ROC) of a Prices Series of Concern: Calculation Function 
def getROC(PriceSeries):
    ROC = ((PriceSeries - PriceSeries.shift(1)) / PriceSeries.shift(1)) * 100
    return ROC

##########################
##########################
#William%R: Calculation Function
def willR(PriceSeries, lookback):
    result = []
    lb = lookback
    for i in range(len(PriceSeries)):
        value = (np.max(PriceSeries[i - lb : i]) - PriceSeries[i]) / (np.max(PriceSeries[i - lb: i]) - np.min(PriceSeries[i - lb : i]))
        result.append(value)
    return result

#########################
#########################   
# Midprice calculation
def midPrice(PriceSeries, lookback):
    result = []
    lb = lookback
    for i in range(len(PriceSeries)):
        value = np.max(PriceSeries[i - lb : i]) - np.min(PriceSeries[i - lb : i])
        result.append(value)
    return result

#########################
#########################
def TAA_Dataset_Transformer(dataset, col_n):
    '''
    Take in relevant Variable/Column from the Stock Market Dataset and use the Techincal Analysis Algorithm (TAA) on it to form the
    Transformed Dataset. For an Undersranding of the Formulae of the TAIs use the Report
    '''
    price_var = dataset[col_n]
    t = 0 ##interation variable for for-loop below
    
    ##Initializing Lists to be Created and Parameters to be used in For-Loop
    History = []
    Dates = dataset.index.tolist()

    EMA = []
    n_EMA = 100 ##for EMA Calcultion
    sc_EMA = 2/(n_EMA+1) #sc = Smoothing Constant
    
    MOM = []
    n_MOM = 10

    BBANDS_mid = []
    BBANDS_upper = []
    BBANDS_lower = []
    Volatil_BBANDS = [] # difference b/w BBANDS_upper and BBANDS_lower 
    hist_BBANDS = []
    beta = 2
    std = 0
    
    for price in price_var:
        
        #For Plotting of Historical Stock Price Data
        History.append(price)
        
        #EMA LIST
        ##Older Prices will be preferred here as they capture Longer-term trends  
        ##and Bollinger Bands/ATR will cover, in a way, short-term "Volatility and Volatile Price Movements"
        if t == 0:
            EMA_val = price
        else:
            EMA_val = (price * sc_EMA) + ((1 - sc_EMA) * EMA_val)
        EMA.append(EMA_val)
        
        #MOMENTUM LIST
        if t >= n_MOM:
            MOM.append(price_var.iloc[t] - price_var.iloc[t - n_MOM])
        else:
            MOM.append(price_var.iloc[t] - price_var.iloc[0]) 
            ##Making a RULE for certain initial prices here
            
        #BBANDS LIST
        BBANDS_mid.append(EMA_val)
        if t < n_EMA:
            hist_BBANDS.append(price)
            std = np.std(hist_BBANDS)
            BBANDS_upper.append(EMA_val + beta*std)
            BBANDS_lower.append(EMA_val - beta*std)
            Volatil_BBANDS.append(2*beta*std)
        else:
            hist_BBANDS = []
            hist_BBANDS.extend(History[t-n_EMA : t])
            std = np.std(hist_BBANDS)
            BBANDS_upper.append(EMA_val + beta*std)
            BBANDS_lower.append(EMA_val - beta*std)            
            Volatil_BBANDS.append(2*beta*std)

        t = t + 1 ##iteration variable is being incremeneted
    
        
    ##Calulating Williams%R, midPrice and ROC separately (outisde main for loop) here
    ROC = getROC(dataset[col_n])
    will_R = willR(dataset[col_n], lookback = 10)
    mid_Price = midPrice(dataset[col_n], lookback = 10)
    
    #Merge the TAIs Together into a DataFrame
    TAI_List_Zipped = list(zip(Dates, EMA, MOM, Volatil_BBANDS, ROC, will_R, mid_Price, History)) 
    TAI_Dataset = pd.DataFrame(TAI_List_Zipped, columns = ["Date", "EMA", "MOM", "Volatility_BBANDS", "ROC", "Will%R", "midPrice", "Prices of Concern"])
    TAI_Dataset = TAI_Dataset.dropna() ##Drop rows with Null Values
    return History, price_var, TAI_Dataset



##################################
##################################
def TAI_Dataset_Preprocessor(dataset, ml_algo):#Take Dataset and ML Algorithm and Preprocess accordingly
    input_scaler = MinMaxScaler(feature_range = (0,1))
    output_scaler = MinMaxScaler(feature_range = (0,1))
    
    if ml_algo == "MLP":
        ##Splitting into Train and Test Sets
        input_train, input_test, output_train, output_test = train_test_split(
    dataset[["EMA", "MOM", "Volatility_BBANDS", "ROC", "Will%R", "midPrice"]], dataset["Prices of Concern"], test_size = 0.1, shuffle = False)
    #Order of Dataset needs to be maintained prior to the Split as it is Sequential.

        ##Scaling Input Sets
        input_train = input_train.values
        input_test = input_test.values
        
        input_train_scaled = input_scaler.fit_transform(input_train)##only learn from training data
        input_test_scaled = input_scaler.transform(input_test)

        ##Scaling the Output Sets 
        output_train = output_train.values.reshape(-1,1)
        output_test = output_test.values.reshape(-1,1)
        
        output_train_scaled = output_scaler.fit_transform(output_train)##only learn from training data
        output_test_scaled = output_scaler.transform(output_test)

        
    #if ml_algo == "LSTM": #univariate_LSTM is the ML Baseline for Deep Learning
        
    #if ml_algo == "RF": #Miscellaneous

    return input_train_scaled, input_test_scaled, output_train_scaled, output_test_scaled, input_scaler, output_scaler ##Returning for Later Denomralization