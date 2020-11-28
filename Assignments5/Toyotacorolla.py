#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Multi-Linear Regression for toyota corolla dataset ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
    


# In[2]:


df = pd.read_csv("C:\\Users\\Home\\Downloads\\ToyotaCorolla.csv",encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df_new = df.drop(["Guarantee_Period","Mfg_Month", "Mfg_Year", "Met_Color","Boardcomputer","Tow_Bar","Radio_cassette","Sport_Model","Metallic_Rim","Mistlamps","CD_Player","Airbag_1","Airbag_2","Mfr_Guarantee","BOVAG_Guarantee","Airco","Automatic_airco","Boardcomputer","Powered_Windows","Power_Steering","Radio","ABS", "Central_Lock", "Backseat_Divider","Fuel_Type", "Color","Id", "Model","Automatic","Cylinders" ], axis = 1) 
  


# In[6]:


df_new.columns


# In[7]:


df_new.shape


# In[8]:


df_new.describe()


# In[9]:


df_new.isnull().sum()


# In[10]:


df_new.hist()


# # df_new.boxplot()   
# # More number of outliers observed in the KM 

# In[11]:


df_new.corr()


# In[12]:


import seaborn as sns


# In[13]:


sns.pairplot(df_new)


# In[14]:


import statsmodels.formula.api as smf


# In[15]:


model1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df_new).fit()


# In[16]:


model1.summary()   # R-Squared value is 0.864 and Doors and cc are insignificant and this model cannot be considered


# In[17]:


model2 = smf.ols('Price ~ Doors + cc', data = df_new).fit()


# In[18]:


model2.summary() # together they are contributing to the model


# In[19]:


model3 = smf.ols('Price~Age_08_04+KM+HP+np.log(cc)+Gears+Quarterly_Tax+Weight', data = df_new).fit()


# In[20]:


model3.summary()  # Rsquared value - 0.867 , Excluded Doors data in this model


# In[21]:


import statsmodels.api as sm


# In[22]:


sm.graphics.influence_plot(model1) # Let us remove 80 row


# In[23]:


df_new1 = df_new.drop(df_new.index[[80]], axis=0)


# In[24]:


model4 = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight+Doors', data = df_new1).fit()


# In[25]:


model4.summary() # After removing 80th row cc is significant and p-value of Doors is decreased


# In[26]:


sm.graphics.influence_plot(model4)


# In[27]:


df_new2 = df_new1.drop(df_new1.index[[221,960]], axis=0)


# In[28]:


model5 = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight+Doors', data = df_new2).fit()


# In[29]:


model5.summary()  # Since is not influencing the signifance level we need to eliminate the column and build the model


# In[30]:


model6 = smf.ols('Price~Age_08_04+KM+HP+cc+Gears+np.log(Quarterly_Tax)+Weight', data = df_new2).fit()


# In[31]:


model6.summary()  # the model is significant and R-squared value is 0.87


# In[36]:


model7 = smf.ols('np.log(Price)~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data = df_new2).fit()


# In[62]:


model7.summary()  # R-sqaured value is lesser than model 7


# In[63]:


rsq_Age_08_04 = smf.ols('Age_08_04~KM+HP+cc+Gears+Quarterly_Tax+Weight', data = df_new2).fit().rsquared
vif_Age_08_04 = 1/(1-rsq_Age_08_04)


# In[64]:


rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Gears+Quarterly_Tax+Weight', data = df_new2).fit().rsquared
vif_KM = 1/(1-rsq_KM)


# In[65]:


rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Gears+Quarterly_Tax+Weight', data = df_new2).fit().rsquared
vif_HP = 1/(1-rsq_HP)


# In[66]:


rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight', data = df_new2).fit().rsquared
vif_cc = 1/(1-rsq_cc)


# In[67]:


rsq_Gears = smf.ols('Gears~Age_08_04+KM+HP+cc+Quarterly_Tax+Weight', data = df_new2).fit().rsquared
vif_Gears = 1/(1-rsq_Gears)


# In[68]:


rsq_Quarterly_Tax = smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Gears+Weight', data = df_new2).fit().rsquared
vif_Quarterly_Tax = 1/(1-rsq_Quarterly_Tax)


# In[69]:


rsq_Weight = smf.ols('Weight~Quarterly_Tax+Age_08_04+KM+HP+cc+Gears', data = df_new2).fit().rsquared
vif_Weight = 1/(1-rsq_Weight)


# In[70]:


VIF = { 'Variables' : ['Weight','Quarterly_Tax','Age_08_04','KM','HP','cc','Gears'], 'Values' : [vif_Weight, vif_Quarterly_Tax,vif_Age_08_04,vif_KM,vif_HP,vif_cc,vif_Gears]}


# In[71]:


VIF_Values = pd.DataFrame(VIF)


# In[72]:


VIF_Values # All the values are less than 5 so multi-collinearity is less


# In[73]:


import statsmodels.api as sm


# In[76]:


sm.graphics.plot_partregress_grid(model7) # Cannot find out as the number of rows are more


# In[ ]:


## Model 6 has the highest r-squared value

