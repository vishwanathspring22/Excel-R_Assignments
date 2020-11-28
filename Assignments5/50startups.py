#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Multi-Linear Regression for 50 startups profit dataset ###
    


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:\\Users\\Home\\Downloads\\50_Startups.csv")


# In[4]:


df.head()


# In[5]:


df_new = df.rename(columns = {'R&D Spend' : 'rdspend', 'Marketing Spend' : 'marketingspend'})


# In[6]:


df['State'].unique()


# In[7]:


df_dc = pd.get_dummies(df_new, columns = ['State'], prefix = '', prefix_sep = '')


# In[8]:


df_dc = df_dc.rename(columns = {'New York' : 'Newyork'})


# In[9]:


df_dc.describe()


# In[10]:


df.isnull().sum() # No null values are present in the dataset


# In[11]:


df.dtypes


# In[12]:


df_dc.hist()


# In[13]:


df_dc.boxplot() # Outliers in profit exist which is not considered as it will be the output


# In[14]:


df.corr()


# In[15]:


import seaborn as sns


# In[16]:


sns.pairplot(df)


# In[17]:


import statsmodels.formula.api as smf


# In[18]:


model1 = smf.ols('Profit~rdspend+Administration+marketingspend+State', data = df_new).fit()


# In[19]:


model1.summary()


# In[20]:


print (model1.conf_int(0.05))


# In[21]:


model2 = smf.ols('Profit~rdspend', data = df_new).fit()


# In[22]:


model2.summary() # R&D spend is significant and adds to the output


# In[23]:


model3 = smf.ols('Profit~Administration', data = df_new).fit()


# In[24]:


model3.summary()  # p-value > 0.05 therefore it is ot significant


# In[25]:


model4 = smf.ols('Profit~State', data = df_new).fit()


# In[26]:


model4.summary()  # Indivisually State has no influence on output as the signifance value is gretare than 0.05


# In[27]:


model5 = smf.ols('Profit~marketingspend', data = df_new).fit()


# In[28]:


model5.summary()  # p-value <0.05, therefore it is significant and it adds to the output 


# In[29]:


model6 = smf.ols('Profit~marketingspend+State', data = df_new).fit()


# In[30]:


model6.summary()  # Even combinely state variable is  not influencing the plot


# In[31]:


import statsmodels.api as sm


# In[32]:


sm.graphics.influence_plot(model1) # Looks like 48, 46, 49 are deviated 


# In[46]:


df_new1 = df_new.drop(df_new.index[[46,48,49,45]], axis=0)
df_new1


# In[34]:


model_new = smf.ols('Profit~rdspend+Administration+marketingspend+State', data = df_new1).fit()


# In[35]:


model_new.summary()


# In[36]:


print(model_new.conf_int(0.01))


# In[37]:


df_pred = model_new.predict(df_new1[['State','rdspend','Administration','marketingspend']])
df_pred


# In[38]:


# Let us calculate the Variance Inflation factor for the variables

rsq_admin = smf.ols('Administration~marketingspend+rdspend+State', data = df_new1).fit().rsquared
vif_admin = 1/(1-rsq_admin)


# In[39]:


rsq_ms = smf.ols('marketingspend~Administration+rdspend+State', data = df_new1).fit().rsquared
vif_ms = 1/(1-rsq_ms)


# In[40]:


rsq_rd = smf.ols('rdspend~marketingspend+Administration+State', data = df_new1).fit().rsquared
vif_rd = 1/(1-rsq_rd)


# In[41]:


# For State variable VIF cannot be calculated as it is not numeric
# if considered with dummy variables it will be nullified by 1.0 as the answer.


# In[42]:


VIF = {'Variables' : ['Administration', 'Marketing Spend', 'R&D Spend'], 'VIF_Value' : [vif_admin, vif_ms, vif_rd]}


# In[43]:


VIF_frame = pd.DataFrame(VIF)


# In[44]:


VIF_frame


# In[45]:


sm.graphics.plot_partregress_grid(model_new)


# In[47]:


# As per observation State variable is very close to 0 so we can remove the column and check for significance with other variables
# Consider a new model without state variable
model7 = smf.ols('Profit~rdspend+Administration+marketingspend', data = df_new1).fit()


# In[48]:


model7.summary()  # Still Administration variable has more p-value


# In[49]:


sm.graphics.plot_partregress_grid(model7) # Administration is nearly 0


# In[92]:


model8 = smf.ols('Profit~rdspend+marketingspend', data = df_new1).fit()


# In[93]:


model8.summary()  # Administration has more p-value so it has to be eliminated 


# In[ ]:


model9 = smf.ols('Profit~marketingspend', data = df_new1).fit()


# In[96]:


model9.summary() # marketing spend indivisaully will contribute for the output


# In[16]:


df_new2 = df_new1.drop(df_new1.index[[36,19,15,14,46]])


# In[102]:


model10 = smf.ols('Profit~rdspend+marketingspend+Administration', data = df_new2).fit()


# In[103]:


model10.summary()


# In[104]:


sm.graphics.influence_plot(model10)


# In[108]:


import numpy as np


# In[129]:


model11 = smf.ols('Profit ~ marketingspend + rdspend', data = df_new2).fit()


# In[123]:


model11.summary()


# In[134]:


df_new2


# In[158]:


df_new2


# In[160]:


df_new3 = df_new2.drop(df_new2.index[[41]])


# In[161]:


df_new3


# In[167]:


df_new3 = df_new2.drop(df_new2.index[[41]])
df_new3


# In[180]:


model12 = smf.ols('Profit ~ marketingspend + rdspend + np.log(Administration)', data = df_new3).fit()


# In[181]:


model12.summary() # Administration is ot significant


# In[192]:


model13 = smf.ols('Profit ~ np.log(marketingspend) + rdspend', data = df_new3).fit()


# In[193]:


model13.summary()


# In[8]:


## r2 value of model10 is higher

RSquared = { 'Model Number' : ['Model1','Model2','Model3','Model4','Model5','Model6','Model_new','Model7','Model8','Model9','Model10','Model11','Model12','Model13'], 'Rsquare-value' : [0.951,0.947,0.040,0.024,0.559,0.562,0.963,0.963,0.961,0.582,0.976,0.974,0.973,0.972] }


# In[10]:


import pandas as pd
RSquarevalue = pd.DataFrame(RSquared)


# In[13]:


RSquarevalue


# In[15]:


RSquarevalue.max()

# 0.976 is the highest

