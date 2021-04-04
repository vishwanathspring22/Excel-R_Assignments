#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Fraud_data = pd.read_csv("C:\\Users\\Home\\Downloads\\Fraud_check.csv")
Fraud_data.columns

Fraud_data['Fraud_Var'] = Fraud_data['Taxable.Income'] <= 30000
Fraud_data.dtypes
Fraud_data['Fraud_Var'].value_counts()

string_col = ['Undergrad','Marital.Status','Urban','Fraud_Var']

#to comvert string fields to numeric
from sklearn import preprocessing
for i in string_col:
    number = preprocessing.LabelEncoder()
    Fraud_data[i] = number.fit_transform(Fraud_data[i])
    
from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:6],train['Fraud_Var'])

preds = model.predict(test.iloc[:,0:6])
pd.Series(preds).value_counts()
pd.crosstab(test['Fraud_Var'],preds)

# Accuracy = train
np.mean(train['Fraud_Var'] == model.predict(train.iloc[:,0:6])) # 1

# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1


#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,0:6],train['Fraud_Var'])
preds = rf.predict(test.iloc[:,0:6])

# Accuracy = train
np.mean(train['Fraud_Var'] == rf.predict(train.iloc[:,0:6])) # 1

# Accuracy = Test
np.mean(preds==test['Fraud_Var']) # 1



########################################################################################



#Decision Tree fraudcheck
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

computer_sale = pd.read_csv("C:\\Users\\Home\\Downloads\\Company_Data.csv")

computer_sale.columns
computer_sale.dtypes

#lets considersales are high if > 7.5, creating the variable to identify sales as high or low
computer_sale['Sale_Var'] = computer_sale['Sales'] >= 7.5
computer_sale.drop("Sales",axis=1,inplace=True)
computer_sale_str_columns = ['ShelveLoc','Urban','US','Sale_Var']

#convert the string columns from numeric
from sklearn import preprocessing
for i in computer_sale_str_columns:
    number = preprocessing.LabelEncoder()
    computer_sale[i] = number.fit_transform(computer_sale[i])
    
from sklearn.model_selection import train_test_split
train,test = train_test_split(computer_sale,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,1:11],train['Sale_Var'])

np.mean(train['Sale_Var'] == model.predict(train.iloc[:,1:11])) #1

np.mean(test['Sale_Var'] == model.predict(test.iloc[:,1:11])) #0.81

#test accuracy is very low oerfit model
#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")

rf.fit(train.iloc[:,1:11],train['Sale_Var'])

np.mean(train['Sale_Var'] == rf.predict(train.iloc[:,1:11])) #0.99

np.mean(test['Sale_Var'] == rf.predict(test.iloc[:,1:11])) #0.95


