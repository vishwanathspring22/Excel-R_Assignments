#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Build a neural network model for predicting profits of the startup.
'''
startup = pd.read_csv("C:\\Users\\Home\\Downloads\\50_Startups.csv")
startup.columns
plt.hist(startup.Profit)

startup.loc[startup.Profit < 105000,"Profit"] = 1 #profit as low
startup.loc[startup.Profit > 105000,"Profit"] = 2 #profit as high

startup.Profit.value_counts()

#to comvert string fields to numeric
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
startup["State"] = number.fit_transform(startup["State"])

X = startup.drop(["Profit"],axis=1)
Y = startup["Profit"]
plt.hist(Y)
startup.Profit.value_counts()

startup.corr()
#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

#applying scale to the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(25,25))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #1
np.mean(y_test==prediction_test) #1

'''
PREDICT THE BURNED AREA OF FOREST FIRES WITH NEURAL NETWORKS
'''

forestfires = pd.read_csv("C:\\Users\\Home\\Downloads\\forestfires.csv")
forestfires.columns
plt.hist(forestfires.size_category)

forestfires.size_category.value_counts()

#converting the category field in numeric
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
forestfires["size_category"] = number.fit_transform(forestfires["size_category"])

#dropping the unnecessary columns
forestfires.drop("month",axis=1,inplace=True)
forestfires.drop("day",axis=1,inplace=True)

X = forestfires.iloc[:,0:28]
Y = forestfires.iloc[:,28]

#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

from sklearn import preprocessing
# standardize the data attributes
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(500,500))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #1
np.mean(y_test==prediction_test) #0.90

'''
Prepare a model for strength of concrete data using Neural Networks
'''
concrete = pd.read_csv("C:\\Users\\Home\\Downloads\\concrete.csv")
concrete.columns
plt.hist(concrete.strength)

concrete.loc[concrete.strength < 30,"strength"] =  0 #weak
concrete.loc[concrete.strength >= 30,"strength"] =  1 #strength

X = concrete.iloc[:,0:8]
Y = concrete.iloc[:,8]

#spliting the data in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y)

from sklearn import preprocessing
# standardize the data attributes
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

from sklearn.neural_network import MLPClassifier
#creating neural network model
mlp = MLPClassifier(hidden_layer_sizes=(100,100))

mlp.fit(X_train,y_train)
prediction_train=mlp.predict(X_train)
prediction_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
np.mean(y_train==prediction_train) #0.97
np.mean(y_test==prediction_test) #0.90

