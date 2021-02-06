#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt


# In[5]:


book = pd.read_csv("C:\\Users\\Home\\Downloads\\book.csv")
frequent_book = apriori(book,min_support=0.005, max_len=3,use_colnames = True)
frequent_book.shape


# In[10]:


# Most Frequent item sets based on support 
frequent_book.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,10)),height = frequent_book.support[1:10],color='rgmyk')
plt.xticks(list(range(1,10)),frequent_book.itemsets[1:10], rotation = 'vertical')
plt.xlabel('item-sets');plt.ylabel('support')


# In[12]:


rules = association_rules(frequent_book, metric="lift", min_threshold=1)
rules.shape


# In[13]:


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules


# In[14]:


def to_list(i):
    return (sorted(list(i)))


# In[15]:


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# In[16]:


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[ ]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt
book = pd.read_csv("C:\\Users\\jzsim\\Downloads\\book.csv")
frequent_book = apriori(book,min_support=0.005, max_len=5,use_colnames = True)
frequent_book.shape

# Most Frequent item sets based on support 
frequent_book.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,10)),height = frequent_movies.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_movies.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_book, metric="lift", min_threshold=1)
rules.shape


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules

def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[17]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from collections import Counter
import matplotlib.pyplot as plt


book = pd.read_csv("C:\\Users\\jzsim\\Downloads\\book.csv")
frequent_book = apriori(book,min_support=0.005, max_len=5,use_colnames = True)
frequent_book.shape

# Most Frequent item sets based on support 
frequent_book.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,10)),height = frequent_movies.support[1:10],color='rgmyk');plt.xticks(list(range(1,10)),frequent_movies.itemsets[1:10])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_book, metric="lift", min_threshold=1)
rules.shape


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules

def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[ ]:




