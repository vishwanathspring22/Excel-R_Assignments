#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pylab as plt


# In[4]:


df = pd.read_csv("C:\\Users\\Home\\Downloads\\crime_data.csv")


# In[5]:


df


# In[6]:


## Normalisation function

def norm_funct(i):
    x =(i - i.min()/ i.max() - i.min())
    return (x)


# In[7]:


# Normalize the numercial part of the data 

df_norm = norm_funct(df.iloc[:,1:])


# In[8]:


from scipy.cluster.hierarchy import linkage


# In[11]:


import scipy.cluster.hierarchy as sch  ## For creating the dendrogram


# In[17]:


df_norm.shape


# In[35]:


df_norm


# In[45]:


## Calculating Eucleadian distance based on linkage " complete"

z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(20,8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0., 
    leaf_font_size = 10
)
plt.show()


# In[46]:


## Applying agglomeratuve clustering choosing 3 clusters from the dendrogram

from sklearn.cluster import AgglomerativeClustering


# In[72]:


h_complete =  AgglomerativeClustering(n_clusters = 3 , linkage = 'complete', affinity = "euclidean").fit(df_norm)


# In[73]:


cluster_labels = pd.Series(h_complete.labels_)


# In[74]:


df['clusters'] = cluster_labels  # Creating new column for cluster labels


# In[80]:


df


# In[81]:


df = df.iloc[:,[4,5,0,1,2,3]]


# In[82]:


df


# In[91]:


#getting aggregate mean and median by grouping each cluster

df.iloc[:,2:].groupby(df.clusters).mean()


# In[92]:


df.iloc[:,2:].groupby(df.clusters).median()


# In[ ]:


## Cluster 0 represents the least murders, assaults, rapes and the least urban population

