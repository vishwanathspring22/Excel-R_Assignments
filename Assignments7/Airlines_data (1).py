#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pylab as plt


# In[3]:


import numpy as np


# In[4]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[5]:


df = pd.read_excel(r"C:\Users\Home\Downloads\EastWestAirlines.xlsx", sheet_name = "data" )


# In[6]:


df.head(40)


# In[7]:


df.isnull().sum()    # No null values are there in the dataset


# In[8]:


# K-Means clustering method

# Normalising Functions

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[9]:


df_norm = norm_func(df.iloc[:,:])


# In[10]:


df_norm.head(10)


# In[11]:


###### screw plot or elbow curve ############

k = list(range(2,15))
k


# In[12]:


TWSS = [] # variable for storing total within sum of squares for each kmeans 


# In[13]:


for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[14]:


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[15]:


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)


# In[16]:


model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust']=md # creating a  new column and assigning it to new column 
df


# In[17]:


df.shape


# In[18]:


df = df.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]


# In[19]:


df.iloc[:,1:12].groupby(df.clust).mean()


# In[20]:


df.iloc[:,1:12].groupby(df.clust).median()


############  Cluster 2 is better when compared to others as flight transactions is more and the number of days enrolled for the flier program is m##e
##  balance to fly more is more ##


# In[21]:


######### Hierarchy clustering method ###################

from scipy.cluster.hierarchy import linkage 


# In[22]:


import scipy.cluster.hierarchy as sch 


# In[23]:


type(df_norm)


# In[24]:


z = linkage(df_norm, method="complete",metric="euclidean")


# In[25]:


plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[26]:


# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from sklearn.cluster import AgglomerativeClustering 


# In[39]:


h_complete = AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(df_norm) 


# In[40]:


cluster_labels=pd.Series(h_complete.labels_)


# In[41]:


df['clust']=cluster_labels # creating a  new column and assigning it to new column 
df


# In[42]:


df.shape


# In[43]:


df= df.iloc[:,[2,0,1,3,4,5,6,7,8,9,10,11,12]]
df.head()


# In[37]:


# getting aggregate mean of each cluster
df.iloc[:,2:].groupby(df.clust).median()


# In[38]:


df.iloc[:,2:].groupby(df.clust).mean()


# In[ ]:


############  Cluster 2 is better when compared to others as flight transactions is more and the number of days enrolled for the flier program is m##e
##  balance to fly more is more ##

