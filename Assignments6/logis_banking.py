# -*- coding: utf-8 -*-
"""logis_banking.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19OUOKG8nTmkHdepAJgtX_5zATRmyN1gg
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install pandas as pd
!pip install numpy as np
!pip install matplotlib
!pip install scipy

import pandas as pd
import numpy as np

df=pd.read_csv('drive/My Drive/logistic/bank-full.csv' , ";")
df

df.shape

#df.describe()

df.isnull().sum()

b

!pip install sklearn

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

df['jobs'] = le.fit_transform(df.job)
print (df.jobs)

df['maritals'] = le.fit_transform(df.marital)
df['educations'] = le.fit_transform(df.education)
df['defaults'] = le.fit_transform(df.default)
df['housings'] = le.fit_transform(df.housing)
df['loans'] = le.fit_transform(df.loan)
df['contacts'] = le.fit_transform(df.contact)
df['months'] = le.fit_transform(df.month)
df['poutcomes'] = le.fit_transform(df.poutcome)
df['yy'] = le.fit_transform(df.y)

df.maritals.nunique()

df

df.drop(['job'],axis=1 ,inplace=True)

!pip install statsmodels

df.columns

import statsmodels.formula.api as sm

logit_model1 = sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+jobs+maritals+educations+defaults+housings+loans+contacts+months+poutcomes', data = df).fit()

logit_model1.summary()

logit_model2 = sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+maritals+educations+defaults+housings+loans+contacts+months+poutcomes', data = df).fit()

logit_model2.summary()

y_pred1 = logit_model2.predict(df)

y_pred1

df['pred1_prob'] = y_pred1

df

df["Att_values"] = 0
df.loc[y_pred1>=0.5, 'Att_values'] = 1
df["Att_values"]

!pip install sklearn

from sklearn.metrics import classification_report

classification_report(df.Att_values, df.yy)

confusion_matrix = pd.crosstab(df['yy'], df.Att_values)
confusion_matrix

accuracy =  ((39139+1137)/(39139+783+1137+4152))
accuracy

## divide data into train and test
df.drop('Att_values', axis =1 , inplace=True)
df

from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size = 0.3)
train.isnull().sum()
test.isnull().sum()
train

train_model = sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+maritals+educations+defaults+housings+loans+contacts+months+poutcomes', data = df).fit()
train_model.summary()

train_pred = train_model.predict(train.iloc[:,:])
print(train)

trains=pd.DataFrame(train)
trains['train_pred']=0
trains.loc[train_pred>=0.5,"train_pred"]=1

trains.head(10)

confusion_matrix=pd.crosstab(trains['yy'],trains.train_pred)
confusion_matrix

accuracy=((27342+812)/(27342+812+559+2934))
accuracy

####Test
test_pred = train_model.predict(test)
test_pred

tests=pd.DataFrame(test)
tests['test_pred']=0
tests.loc[test_pred>=0.5,"test_pred"]=1

testmatrix = pd.crosstab(tests['yy'], tests.test_pred)
testmatrix

accuracy = ((11797+325)/(11797+224+325+1218))
accuracy