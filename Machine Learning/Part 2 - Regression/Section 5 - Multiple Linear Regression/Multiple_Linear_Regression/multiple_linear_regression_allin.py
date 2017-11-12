# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset from working directory
dataset= pd.read_csv('50_Startups.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]= labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x = onehotencoder.fit_transform(x).toarray()

#Avoid Dummy variable trap by deleting 1 coloums
x=x[:, 1:]

#splitting training and test data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predict results
y_pred=regressor.predict(x_test)

plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.title('Actual results vs Predicted results')
plt.xlabel('Independent variables represented by numbers')
plt.ylabel('Dependent variabe profit')