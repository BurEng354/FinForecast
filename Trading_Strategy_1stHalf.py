# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:14:24 2022

@author: burhan
"""


import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout, Input, Flatten
from keras.models import Sequential
from keras import callbacks

##Preparing Subplot Objects
MLP_fig, (MLP_train_subplt, MLP_predicts_subplt) = plt.subplots(2,1)
MLP_fig.suptitle("AI Financial Forecaster on MLP")

num_TAIs = 6
#Tip used: Overfitting the MLP first and then Tuning it (to Justify Architecture)
model_MLP = Sequential()
model_MLP.add(Flatten(input_shape=(num_TAIs, )) ) # we need to specify the input shape or we won't be able to see the summary
model_MLP.add(Dense(80, activation='relu'))
#model_MLP.add(Dropout(0.2)) #To prevent Overfitting
model_MLP.add(Dense(40, activation='relu'))
#model_MLP.add(Dropout(0.2))
model_MLP.add(Dense(20, activation='relu'))
model_MLP.add(Dense(1, activation='sigmoid'))
model_MLP.summary()

##Using Adaptive Learning Rate to circumvent having to try different Learning Rates in non-Adaptive Mechanisms
model_MLP.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

def MLP_model(MLP_modality, TAIinput_train_scaled, TAIoutput_train_scaled, TAIinput_test_scaled, op_scalar, TA_Indicators_Dataset):
    if MLP_modality == "Train": 
        history_MLP = model_MLP.fit(TAIinput_train_scaled, TAIoutput_train_scaled, epochs = 100, validation_split = 0.222)

        MLP_train_subplt.plot(history_MLP.history['loss'], label='Training Error')
        MLP_train_subplt.plot(history_MLP.history['val_loss'], label = 'Validation Error')
        MLP_train_subplt.set_xlabel('Epochs')
        MLP_train_subplt.set_ylabel('Loss/MSE')
        MLP_train_subplt.legend(loc='upper right')
        
    if MLP_modality == "Predict":
        input_scaled = np.concatenate((TAIinput_train_scaled, TAIinput_test_scaled))
        predictions_scaled_MLP = model_MLP.predict(input_scaled)
        predictions_scaled_MLP = predictions_scaled_MLP.reshape(-1,1)
        predictions_MLP = op_scalar.inverse_transform(predictions_scaled_MLP)
        
        MLP_predicts_subplt.plot(predictions_MLP, label='Predicted Closings Prices of MLP')
        MLP_predicts_subplt.plot(TA_Indicators_Dataset['Prices of Concern'].values, label = 'True CLosing Prices')
        MLP_predicts_subplt.set_xlabel('Timesteps from chosen Start Date')
        MLP_predicts_subplt.set_ylabel('Stock Price of Concern')
        MLP_predicts_subplt.legend(loc='upper right')
    
    print("Saving the Subplots corresponding to MLP")
    
    MLP_fig.tight_layout()
    MLP_fig.savefig("MLP_fig.jpg")
    return

'''
Recommendations:
    
1) Incorporate Model Saving
2) Incorporate OOP
'''