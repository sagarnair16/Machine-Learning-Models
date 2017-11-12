# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:46:36 2017

@author: Sagar
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset from working directory
dataset= pd.read_csv('Salary_Data.csv')

x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, 1].values

#splitting training and test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =1/3, random_state=0)


#Fitting Simple linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the results
y_pred = regressor.predict(x_test)

#Visualize the training set results
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualize the test set results
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()