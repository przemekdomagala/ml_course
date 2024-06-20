# %% [markdown]
# # Wczytanie danych

# %%
from sklearn import datasets
from sklearn.datasets import load_iris

data_breast_cancer = datasets.load_breast_cancer()
data_iris = load_iris()

# %% [markdown]
# # Zadanie 1

# %%
from sklearn.decomposition import PCA
import numpy as np

X_breast_cancer = data_breast_cancer.data
y_breast_cancer = data_breast_cancer.target

X_iris = data_iris.data
y_iris = data_iris.target

# determiting components for breast_cancer
pca = PCA()
pca = PCA(n_components=0.9)
X_breast_cancer_reduced = pca.fit_transform(X_breast_cancer)
print(pca.explained_variance_ratio_)

# determiting components for iris
pca = PCA()
pca = PCA(n_components=0.9)
X_breast_cancer_reduced = pca.fit_transform(X_iris)
print(pca.explained_variance_ratio_)

# %% [markdown]
# # Zadanie 2, 3

# %%
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
X_breast_scaled = scaler.fit_transform(X_breast_cancer)
X_iris_scaled = scaler.fit_transform(X_iris)

pca_bc_scaled = PCA(n_components=0.9)
pca_bc_scaled.fit_transform(X_breast_scaled)
print(pca_bc_scaled.explained_variance_ratio_)

with open('pca_bc.pkl', 'wb') as pca_bc_file:
    pickle.dump(pca_bc_scaled.explained_variance_ratio_, pca_bc_file)

pca_iris_scaled = PCA(n_components=0.9)
pca_iris_scaled.fit_transform(X_iris_scaled)
print(pca_iris_scaled.explained_variance_ratio_)

with open('pca_ir.pkl', 'wb') as pca_ir_file:
    pickle.dump(pca_iris_scaled.explained_variance_ratio_, pca_ir_file)

# %% [markdown]
# # Zadanie 4

# %%
idx_bc = np.argmax(np.abs(pca_bc_scaled.components_), axis=1)
idx_ir = np.argmax(np.abs(pca_iris_scaled.components_), axis=1)

with open('idx_bc.pkl', 'wb') as idx_bc_file:
    pickle.dump(idx_bc, idx_bc_file)

with open('idx_ir.pkl', 'wb') as idx_ir_file:
    pickle.dump(idx_ir, idx_ir_file)


