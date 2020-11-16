#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


delivery_time = pd.read_csv("C:\\Users\\Home\\Downloads\\delivery_time.csv")


# In[6]:


delivery_time


# In[22]:


delivery_time.hist(column='Delivery Time')


# In[25]:


delivery_time.boxplot(column='Delivery Time')


# In[27]:


delivery_time.isnull().sum()


# In[30]:


delivery_time.describe()


# In[31]:


delivery_time.boxplot(column='Sorting Time')


# In[32]:


delivery_time.hist(column='Sorting Time')


# In[38]:


delivery_time.rename(columns = {'Delivery Time':'DT', 'Sorting Time': 'ST'}, inplace = True) 
   


# In[40]:


delivery_time


# In[42]:


plt.hist(delivery_time.DT)


# In[44]:


plt.hist(delivery_time.ST)


# In[45]:


plt.boxplot(delivery_time.ST)


# In[46]:


plt.boxplot(delivery_time.DT)


# In[47]:


plt.plot(delivery_time.ST,delivery_time.DT,"bo");plt.xlabel("ST");plt.ylabel("DT")


# In[48]:


delivery_time.plot.scatter(x = 'ST', y = 'DT')
# Two methods of plotting a scatter plot


# In[50]:


delivery_time.DT.corr(delivery_time.ST)


# In[52]:


import statsmodels.formula.api as smf


# In[98]:


model = smf.ols("DT ~ ST", data = delivery_time).fit()


# In[99]:


model.params


# In[100]:


model.summary()


# In[79]:


print (model.conf_int(0.05))


# In[81]:


delivery_time


# In[95]:


model


# In[101]:


pred = model.predict(delivery_time.ST)


# In[103]:


pred # Predicted values


# In[104]:


import matplotlib.pyplot as plt


# In[116]:


plt.scatter(x = delivery_time['ST'], y = delivery_time['DT'], color = 'blue')
plt.plot(delivery_time['ST'],pred,color = 'red')
plt.xlabel("ST")
plt.ylabel("DT")


# In[112]:


pred.corr(delivery_time.DT)


# In[118]:


model2 = smf.ols('DT~np.log(ST)', data = delivery_time).fit()


# In[120]:


model2.params


# In[122]:


model2.summary() # not significant as p>0.05


# In[124]:


pred2 = model2.predict(delivery_time.ST)


# In[126]:


pred2


# In[128]:


model3 = smf.ols('np.log(DT)~(ST)', data = delivery_time).fit()


# In[130]:


model3.params


# In[132]:


model3.summary()


# In[135]:


model3.conf_int(0.01)


# In[137]:


pred_log = model3.predict(delivery_time.ST)


# In[140]:


pred3 = np.exp(pred_log)


# In[142]:


pred3


# In[151]:


plt.scatter(x=delivery_time['ST'], y = delivery_time['DT'], color= 'blue')
plt.plot(delivery_time['ST'],pred3, color = 'red')
plt.xlabel("ST")
plt.ylabel("DT")


# In[153]:


pred3.corr(delivery_time.DT)


# In[155]:


student_resid = model3.resid_pearson


# In[157]:


student_resid


# In[159]:



plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# In[161]:


plt.scatter(x=pred3,y=delivery_time.DT);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[179]:


# Quadratic model
delivery_time["ST_Sq"] = delivery_time.ST*delivery_time.ST


# In[192]:


model_quad = smf.ols("np.log(DT)~ST+ST_Sq",data=delivery_time).fit()


# In[193]:


model_quad.params


# In[194]:


model_quad.summary()


# In[199]:


# p>0.05  cannot be considered


# In[ ]:


# deivery time taken exponential has more r-sqaured value

