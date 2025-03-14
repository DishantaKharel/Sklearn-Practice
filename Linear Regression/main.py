from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd



california = datasets.fetch_california_housing()


X = california.data
y = california.target


#Algorithm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[2], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Prediction:", predictions)
print(y_test)

#Checks the accuracy
print("R^2 value:", l_reg.score(X,y))

#Show how much each feature affects the price.
print("Coeff:", l_reg.coef_)
print("intercept:",l_reg.intercept_)