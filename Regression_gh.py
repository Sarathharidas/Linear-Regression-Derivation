# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 01:36:33 2019

@author: sarat
"""

import random
import pandas as pd 
import numpy as np
from math import sqrt
from statistics import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mse
data = pd.read_csv('C:/Users/sarat/Desktop/Spring/Causal Inference/BMI.csv')
data1 = data[['weight','height']]

## Using basic statistics
x_bar = data1.height.mean()
y_bar = data1.weight.mean()
            
num = sum((data1.height - x_bar)*(data1.weight-y_bar))
den = sum((data1.height - x_bar)*(data1.height - x_bar))

b1 = num/den

b0 = y_bar - b1*x_bar

y_pred = b1*data1.height +b0

rmse = sqrt(mean((y_pred - data1.weight)*(y_pred - data1.weight)))
## RMSE value is 40.8 

###now lets use sklearn 
model =LinearRegression()
x=np.array(data1.height)
y = np.array(data1.weight)

# Reshaping data to put in linear regression
x=x.reshape(-1,1)
y=y.reshape(-1,1)

model.fit(x,y)
y_pred = model.predict(x)
rmse = sqrt(mse(y_pred,y))
rmse 
# 40.8 
# the values from both methods match 