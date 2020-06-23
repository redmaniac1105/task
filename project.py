# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('datafile (1).csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in range(0,2):
    X[:, i] = labelencoder.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [i])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Random Forest Regression
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor(n_estimators = 740, random_state = 0)
RFregressor.fit(X_train, y_train)
# Predicting results
y_pred=RFregressor.predict(X_test)
#Calculating MSE
j=[]
mse=[]
for i in range(0,10):
    j.insert(i,(y_pred[i]-y_test[i])**2)
mse.insert(0,sum(j))
#Calculating Accuracy
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = RFregressor, X = X_train, y = y_train, cv = 4)
accuracies.mean()
accuracies.std()



#Simple Linear Regression
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
Lregressor = LinearRegression()
Lregressor.fit(X_train, y_train)
# Predicting results
y_pred = Lregressor.predict(X_test)
#Calculating MSE    
for i in range(0,10):
    j.insert(i,(y_pred[i]-y_test[i])**2)
mse.insert(1,sum(j))
#Calculating Accuracy
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = Lregressor, X = X_train, y = y_train, cv = 4)
accuracies.mean()
accuracies.std()



#Polynomial regression
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
Lregressor.fit(X_poly, y_train)
#Predicting results
y_pred = Lregressor.predict(poly_reg.fit_transform(X_test))
#Calculating MSE
for i in range(0,10):
    j.insert(i,(y_pred[i]-y_test[i])**2)
mse.insert(2,sum(j))
#Calculating Accuracy
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = RFregressor, X = X_train, y = y_train, cv = 4)
accuracies.mean()
accuracies.std()


#SVM Regression
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
y_train = np.array(y_train).reshape(-1,1)
y_train = sc_y.fit_transform(y_train)
# Fitting SVR to the dataset
from sklearn.svm import SVR
SVregressor = SVR(kernel = 'rbf')
SVregressor.fit(X_train, y_train)
# Predicting a new result
y_pred = SVregressor.predict(X_test)
#Calculating Accuracy
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator = SVregressor, X = X_train, y = y_train, cv = 4)
accuracies.mean()
accuracies.std()



