#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlretrieve
import tarfile
import gzip
import os
import shutil
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
filename = "housing.tgz"
urlretrieve(url, filename)
with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()


# In[2]:


extracted_file = "housing.csv"
compressed_file = "housing.csv.gz"
with open(extracted_file, 'rb') as file_in, gzip.open(compressed_file, 'wb') as file_out:
    shutil.copyfileobj(file_in, file_out)


# In[3]:


import pandas as pd
df = pd.read_csv('housing.csv.gz')


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


print(df['ocean_proximity'].dtypes)
print(df['ocean_proximity'].value_counts())
df['ocean_proximity'].describe()


# In[7]:


histogram = df.hist(bins=50, figsize=(20,15))
fig = histogram[0][0].get_figure()
fig.savefig("obraz1.png")


# In[8]:


plot2 = df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7,4))
fig2 = plot2.get_figure()
fig2.savefig("obraz2.png")


# In[9]:


import matplotlib.pyplot as plt
plot3 = df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(7,3), colorbar=True, s=df["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"))
fig3 = plot3.get_figure()
fig3.savefig("obraz3.png")


# In[10]:


corr_matrix = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
corr_matrix_df = corr_matrix.reset_index()
corr_matrix_df.columns = ['atrybut', 'wspolczynnik_korelacji']
corr_matrix_df.to_csv('korelacja.csv', index=False)


# In[11]:


import seaborn as sns
sns.pairplot(df)


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)
print(train_set)
print(test_set)
train_set_corr = train_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
print(train_set_corr)
test_set_corr = test_set.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
print(test_set_corr)


# In[13]:


import pickle

with open('train_set.pkl', 'wb') as handle:
    pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_set.pkl', 'wb') as handle:
    pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




