# %% [markdown]
# # Przygotowanie danych

# %%
from sklearn.datasets import fetch_openml
import numpy as np
import pickle

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

# %% [markdown]
# # Zadanie 1

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

lst1 = []

# %%
kmeans_8 = KMeans(n_clusters=8, n_init=10)
sylwetkowy8 = silhouette_score(X, kmeans_8.fit_predict(X))
lst1.append(sylwetkowy8)

# %%
kmeans_9 = KMeans(n_clusters=9, n_init=10)
sylwetkowy9 = silhouette_score(X, kmeans_9.fit_predict(X))
lst1.append(sylwetkowy9)

# %%
kmeans_10 = KMeans(n_clusters=10, n_init=10)
sylwetkowy10 = silhouette_score(X, kmeans_10.fit_predict(X))
lst1.append(sylwetkowy10)

# %%
kmeans_11 = KMeans(n_clusters=11, n_init=10)
sylwetkowy11 = silhouette_score(X, kmeans_11.fit_predict(X))
lst1.append(sylwetkowy11)

# %%
kmeans_12 = KMeans(n_clusters=12, n_init=10)
sylwetkowy12 = silhouette_score(X, kmeans_12.fit_predict(X))
lst1.append(sylwetkowy12)

# %%
with open('kmeans_sil.pkl', 'wb') as kmeans_file:
    pickle.dump(lst1, kmeans_file)

lst1

# %% [markdown]
# # Zadanie 4

# %%
import pandas as pd

from sklearn.metrics import confusion_matrix

m = confusion_matrix(y, kmeans_10.predict(X))
set_ = set()
for i in m:
    set_.add(i.argmax())

set_ = sorted(set_)

with open("kmeans_argmax.pkl", 'wb') as arg_file:
    pickle.dump(set_, arg_file)

print(set_)

# %% [markdown]
# # Task 6

# %%
from sklearn.cluster import DBSCAN

picklelist = []
distances = []
for i in X[:300]:
    dist = np.linalg.norm(X - i, axis=1)
    distances.append(dist)
distances = np.array(distances)
np.fill_diagonal(distances, np.inf)
smallest_indices = np.argsort(distances, axis=None)
for i in range(10):
    idx1, idx2 = np.unravel_index(smallest_indices[i], distances.shape)
    picklelist.append(distances[idx1, idx2])

with open("dist.pkl", 'wb') as dst_file:
    pickle.dump(picklelist, dst_file)

picklelist

# %%
avg = distances[np.unravel_index(smallest_indices[0], distances.shape)]+ distances[np.unravel_index(smallest_indices[1], distances.shape)]+ distances[np.unravel_index(smallest_indices[2], distances.shape)]
avg /= 3
avg
eps1 = avg
eps2 = avg+0.04*avg
eps3 = avg+0.08*avg

# %%
dbscan1 = DBSCAN(eps=eps1)
dbscan1.fit(X)

dbscan2 = DBSCAN(eps=eps2)
dbscan2.fit(X)

dbscan3 = DBSCAN(eps=eps3)
dbscan3.fit(X)


# %%
dbscan_lst = []
dbscan_lst.append(len(np.unique(dbscan1.labels_)))
dbscan_lst.append(len(np.unique(dbscan2.labels_)))
dbscan_lst.append(len(np.unique(dbscan3.labels_)))

with open("dbscan_len.pkl", 'wb') as dbscan_file:
    pickle.dump(dbscan_lst, dbscan_file)

dbscan_lst


