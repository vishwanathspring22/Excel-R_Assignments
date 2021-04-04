#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import re

#cocacola use case
#from sm.tsa.statespace import sa
CocaCola_Sales = pd.read_excel("C:\\Users\\Vishwa\\CocaCola_Sales_Rawdata.xlsx")
plt.plot(CocaCola_Sales["Quarter"],CocaCola_Sales["Sales"],linewidth=2.0)

#As data is quaterly so we need to convert it in 4 dummy variable
lst = []
for i in CocaCola_Sales["Quarter"]:
    x = re.sub('Q1','Quater_1',i)
    y = re.sub('Q2','Quater_2',x)
    z = re.sub('Q3','Quater_3',y)
    d = re.sub('Q4','Quater_4',z)
    lst.append(d)


col = []
for j in lst:
    x = j[0:8]
    col.append(x)

CocaCola_Sales['forcast'] = col
CocaCola_Sales =  CocaCola_Sales.drop("Quarter",axis=1)


quater_dummies = pd.DataFrame(pd.get_dummies(CocaCola_Sales['forcast']))
CocaCola_Sales = pd.concat([CocaCola_Sales,quater_dummies],axis = 1)

x = CocaCola_Sales.shape[0]
CocaCola_Sales["t"] = np.arange(1,43)

CocaCola_Sales["t_squared"] = CocaCola_Sales["t"]*CocaCola_Sales["t"]

CocaCola_Sales["log_Sales"] = np.log(CocaCola_Sales["Sales"])
Train = CocaCola_Sales.head(36)
Test = CocaCola_Sales.tail(4)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
CocaCola_Sales.columns
################### Additive seasonality ########################

add_sea = smf.ols('Sales~Quater_1+Quater_2+Quater_3+Quater_4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Quater_1','Quater_2','Quater_3','Quater_4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Quater_1+Quater_2+Quater_3+Quater_4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Quater_1','Quater_2','Quater_3','Quater_4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales~Quater_1+Quater_2+Quater_3+Quater_4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sales~t+Quater_1+Quater_2+Quater_3+Quater_4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#Airlines use case
airlines = pd.read_excel("C:\\Users\\Vishwa\\Airlines+Data.xlsx")
airlines.dtypes

airlines["month"] = airlines.Month.dt.strftime("%b")

month_dummies = pd.DataFrame(pd.get_dummies(airlines['month']))
airlines = pd.concat([airlines,month_dummies],axis = 1)
airlines.shape

airlines["t"] = np.arange(1,97)

airlines["t_squared"] = airlines["t"]*airlines["t"]
airlines.columns

airlines["log_Rider"] = np.log(airlines["Passengers"])
airlines.Passengers.plot()

Train = airlines.head(84)
Test = airlines.tail(12)

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Rider~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Rider~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Rider~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#Plastic Sales use case
Plastic_Sales = pd.read_csv("C:\\Users\\Vishwa\\PlasticSales.csv")
plt.plot(Plastic_Sales["Month"],Plastic_Sales["Sales"],linewidth=2.0)

Plastic_Sales['Month']
Plastic_Sales.shape

lst = []
for i in Plastic_Sales.Month:
    lst.append(re.sub("[^A-Za-z""]+"," ",i).lower())
l=[]
for i in lst:
    l.append(re.sub(" ","",i).lower())

Plastic_Sales['month'] = l
month_dummies = pd.DataFrame(pd.get_dummies(Plastic_Sales['month']))
Plastic_Sales = pd.concat([Plastic_Sales,month_dummies],axis = 1)

Plastic_Sales["t"] = np.arange(1,61)

Plastic_Sales["t_squared"] = Plastic_Sales["t"]*Plastic_Sales["t"]

Plastic_Sales["log_Rider"] = np.log(Plastic_Sales["Sales"])
Plastic_Sales.Sales.plot()
Train = Plastic_Sales.head(48)
Test = Plastic_Sales.tail(12)    



####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_Rider~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales~jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Rider~jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Rider~t+jan+feb+mar+apr+may+jun+jul+aug+sep+oct+nov+dec',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

