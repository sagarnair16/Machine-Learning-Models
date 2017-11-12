# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# fitting Linear regression model to the data
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(x,y)

# fitting polynomial regression into the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualize the linear regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize the polynomial regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predict result with linear regression
lin_reg.predict(6.5)

#Predict result with Polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))