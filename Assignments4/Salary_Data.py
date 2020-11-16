#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


salary_data = pd.read_csv("C:\\Users\\Home\\Downloads\\Salary_Data.csv")


# In[7]:


salary_data

# input is experience and output is salary


# In[19]:


salarydata = salary_data.rename(columns = {'YearsExperience' : 'exp'})


# In[20]:


salarydata


# In[22]:


salarydata.describe()


# In[25]:


salarydata.shape


# In[26]:


salarydata.hist(column = 'exp' )


# In[28]:


salarydata.boxplot(column = 'exp' )


# In[29]:


salarydata.isnull().sum()


# In[30]:


salarydata.boxplot(column = 'Salary' )


# In[31]:


salarydata.hist(column = 'Salary' )


# In[48]:


plt.plot(salarydata.exp,salarydata.Salary, "bo")
plt.xlabel("Experience")
plt.ylabel("Salary")


# In[49]:


salarydata.Salary.corr(salarydata.exp)


# In[50]:


import statsmodels.formula.api as smf


# In[51]:


model = smf.ols('Salary ~ exp',data = salarydata).fit()


# In[53]:


model.params


# In[54]:


model.summary()


# In[56]:


model.conf_int(0.01)


# In[57]:


pred1 =  model.predict(salarydata.exp)


# In[68]:


plt.scatter(x = salarydata['exp'], y = salarydata['Salary'], color = 'blue')
plt.plot(salarydata['exp'], pred1 , color = 'red')
plt.xlabel("Experience")
plt.ylabel("Salary")


# In[69]:


pred1.corr(salarydata.exp)


# In[ ]:


# It is perfectly correlated

