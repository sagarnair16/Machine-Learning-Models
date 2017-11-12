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


#add 1 coloum at the end of your dataset fro backward elimination.
import statsmodels.formula.api as sm
x = np.append(arr= np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# removing coloum 2 since it's p>sl (sl=0.05)
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# removing coloum 1 since it's p>sl (sl=0.05)
x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# removing coloum 4 since it's p>sl (sl=0.05)
x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# removing coloum 5 since it's p>sl (sl=0.05)
x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y , exog = x_opt).fit()
regressor_OLS.summary()

# Now we know there is a strong co-relation between coloum 3 which is R&D and Profit.

#Find linear regression between R&D and Profit
# Convert integer to matrix since that is needed for regressor to fit to models.
x=x[:, 3:4]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size =0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Prediction
y_pred = regressor.predict(x_test)

# Plot predicted and actual values
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='blue')
plt.title('Actual vs Predicted for R&D vs Profit')
plt.xlabel('R&D spend')
plt.ylabel('Profit')
plt.show()

