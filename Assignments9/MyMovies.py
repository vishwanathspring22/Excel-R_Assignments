#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt


# In[23]:


df = pd.read_csv("C:/Users/Home/Downloads/my_movies (1).csv")
df


# In[24]:


df.isnull().sum() 


# In[26]:


df
df = df.iloc[ : , 5 :]


# In[27]:


df


# In[30]:


frequent_movies = apriori(df,min_support=0.005, max_len=3,use_colnames = True)
frequent_movies.shape


# In[33]:


# Most Frequent item sets based on support 
frequent_movies.sort_values('support',ascending = False,inplace=True)


# In[41]:


plt.bar(x = list(range(1,10)),height = frequent_movies.support[1:10],color='rgmyk')
plt.xticks(list(range(1,10)),frequent_movies.itemsets[1:10], rotation = 'vertical')
plt.xlabel('item-sets');plt.ylabel('support')


# In[43]:


rules = association_rules(frequent_movies, metric="lift", min_threshold=1)
rules.shape


# In[44]:


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules


# In[45]:


def to_list(i):
    return (sorted(list(i)))


# In[46]:


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


# In[47]:



ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# In[48]:


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[ ]:


##################################################################

