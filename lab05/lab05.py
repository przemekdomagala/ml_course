# %%
from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])

# %%
import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')

# %% [markdown]
# # Klasyfikacja

# %%
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error

X_breast_cancer = data_breast_cancer.data[["mean texture", "mean symmetry"]]
y_breast_cancer = data_breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer, y_breast_cancer, test_size=0.2, random_state=42)

# print("### DEPTH 2 ###")
# tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# tree_clf.fit(X_train, y_train)
# print(f1_score(y_train, tree_clf.predict(X_train)))
# print(f1_score(y_test, tree_clf.predict(X_test)))

# print("\n### DEPTH 3 ###")
# tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# tree_clf.fit(X_train, y_train)
# print(f1_score(y_train, tree_clf.predict(X_train)))
# print(f1_score(y_test, tree_clf.predict(X_test)))

# print("\n### DEPTH 4 ###")
# tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
# tree_clf.fit(X_train, y_train)
# print(f1_score(y_train, tree_clf.predict(X_train)))
# print(f1_score(y_test, tree_clf.predict(X_test)))

# print("\n### DEPTH 5 ###")
# tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
# tree_clf.fit(X_train, y_train)
# print(f1_score(y_train, tree_clf.predict(X_train)))
# print(f1_score(y_test, tree_clf.predict(X_test)))

### After depth 3 f1 for test set is getting lower, so I am staying with depth 3

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)
print(f1_score(y_train, tree_clf.predict(X_train)))
print(f1_score(y_test, tree_clf.predict(X_test)))

# %% [markdown]
# # Generowanie obrazka

# %%
from sklearn.tree import export_graphviz

export_graphviz(tree_clf, out_file="bc.dot", feature_names=["mean texture", "mean symmetry"], class_names=data_breast_cancer.target_names, rounded=True, filled=True)

from graphviz import Source

Source.from_file("bc.dot")

# %%
#!dot -Tpng {"bc.dot"} -o {"bc.png"}

import subprocess

# Execute the dot command
subprocess.run(["dot", "-Tpng", "bc.dot", "-o", "bc.png"])

# %% [markdown]
# # Lista

# %%
lst = []
lst.append(tree_clf.get_depth())
lst.append(f1_score(y_train, tree_clf.predict(X_train)))
lst.append(f1_score(y_test, tree_clf.predict(X_test)))
lst.append(accuracy_score(y_train, tree_clf.predict(X_train)))
lst.append(accuracy_score(y_test, tree_clf.predict(X_test)))

import pickle

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(lst, f)

print(lst)

# %% [markdown]
# # Zadanie 2

# %%
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# print("### DEPTH 2 ###")
# tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg.fit(X_train, y_train)
# print(mean_squared_error(y_train, tree_reg.predict(X_train)))
# print(mean_squared_error(y_test, tree_reg.predict(X_test)))

# print("\n### DEPTH 3 ###")
# tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
# tree_reg.fit(X_train, y_train)
# print(mean_squared_error(y_train, tree_reg.predict(X_train)))
# print(mean_squared_error(y_test, tree_reg.predict(X_test)))

print("\n### DEPTH 4 ###")
tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg.fit(X_train, y_train)

mse_train = mean_squared_error(y_train, tree_reg.predict(X_train))
mse_test = mean_squared_error(y_test, tree_reg.predict(X_test))

# print("\n### DEPTH 5 ###")
# tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
# tree_reg.fit(X_train, y_train)
# print(mean_squared_error(y_train, tree_reg.predict(X_train)))
# print(mean_squared_error(y_test, tree_reg.predict(X_test)))

# print("\n### DEPTH 6 ###")
# tree_reg = DecisionTreeRegressor(max_depth=6, random_state=42)
# tree_reg.fit(X_train, y_train)
# print(mean_squared_error(y_train, tree_reg.predict(X_train)))
# print(mean_squared_error(y_test, tree_reg.predict(X_test)))

# %% [markdown]
# # WYKRES

# %%
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')
plt.scatter(tree_reg.predict(X_train), y_train, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('max_depth=4')
plt.legend()
plt.grid(True)
plt.show()

# %%
export_graphviz(tree_reg, out_file="reg.dot", feature_names=["x1"], rounded=True, filled=True)
Source.from_file("reg.dot")

# %%
# !dot -Tpng {"reg.dot"} -o {"reg.png"}

import subprocess

# Execute the dot command
subprocess.run(["dot", "-Tpng", "reg.dot", "-o", "reg.png"])


# %%
lst2 = []
lst2.append(tree_reg.get_depth())
lst2.append(mse_train)
lst2.append(mse_test)

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(lst2, f)

print(lst2)


