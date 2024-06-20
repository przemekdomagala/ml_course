# %%
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=True, version=1)
print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))

# %%
X, y = mnist.data, mnist.target
y = y.sort_values(ascending=True)
X = X.reindex(index=y.index)
X_train, X_test, y_train, y_test = X[:56000], X[56000:], y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_train.unique())
print(y_test.unique())

# %%
from sklearn.model_selection import train_test_split
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print(y_train.unique())
print(y_test.unique())

# %%
y_train_0 = (y_train == '0')
y_test_0 = (y_test == '0')
#print(y_test_0.unique())

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)
print(mnist["data"].loc[1:1])
sgd_clf.predict(mnist["data"].loc[0:0])
sgd_clf.predict(mnist["data"].loc[1:1])

# %%
y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)

acc_train = sum(y_train_pred == y_train_0)/len(y_train_0)
acc_test = sum(y_test_pred == y_test_0)/len(y_test_0)

print(acc_train, acc_test)
lst_float = [acc_train, acc_test]
print(type(lst_float))

import pickle

with open('sgd_acc.pkl', 'wb') as file:
    pickle.dump(lst_float, file)
    
from sklearn.model_selection import cross_val_score
cross_test = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")
print(type(cross_test))

with open('sgd_cva.pkl', 'wb') as file_nd:
    pickle.dump(cross_test, file_nd)

# %%
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])
svm_clf.predict(mnist["data"].loc[0:0])
print(svm_clf.classes_)
y_train_pred = cross_val_predict(svm_clf, X_train, y_train, cv=3, n_jobs=-1)

# %%
# M = ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# print(M)
from sklearn import metrics 
confusion_matrix = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion_matrix)

# %%
# arr = M.text_
# print(arr)
print(type(confusion_matrix))
print(confusion_matrix.shape)
with open('sgd_cmx.pkl', 'wb') as file_rd:
    pickle.dump(confusion_matrix, file_rd)


