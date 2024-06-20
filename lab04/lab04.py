# %%
from sklearn import datasets

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])

# %%
# data_iris = datasets.load_iris()
data_iris = datasets.load_iris(as_frame=True)
print(data_iris["DESCR"])

# %%
from sklearn.model_selection import train_test_split

X = data_breast_cancer["data"][["mean area", "mean smoothness"]]
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

model1 = LinearSVC(loss='hinge', random_state=42)
model1.fit(X_train, y_train)

model2 = make_pipeline(StandardScaler(), LinearSVC(loss='hinge', random_state=42))
model2.fit(X_train, y_train)

acc_bc = [accuracy_score(y_train, model1.predict(X_train)), accuracy_score(y_test, model1.predict(X_test)), accuracy_score(y_train, model2.predict(X_train)), accuracy_score(y_test, model2.predict(X_test))]

with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(acc_bc, file)

print(acc_bc)

# %%
X_iris = data_iris['data'][['petal length (cm)', 'petal width (cm)']]
y_iris = data_iris.target_names[data_iris.target] == 'virginica'
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

model1 = LinearSVC(loss='hinge', random_state=42)
model1.fit(X_iris_train, y_iris_train)

model2 = make_pipeline(StandardScaler(), LinearSVC(loss='hinge', random_state=42))
model2.fit(X_iris_train, y_iris_train)

iris_acc = [accuracy_score(y_iris_train, model1.predict(X_iris_train)), accuracy_score(y_iris_test, model1.predict(X_iris_test)), 
            accuracy_score(y_iris_train, model2.predict(X_iris_train)), accuracy_score(y_iris_test, model2.predict(X_iris_test))]

with open('iris_acc.pkl', 'wb') as file:
    pickle.dump(iris_acc, file)

print(iris_acc)

# %%
import numpy as np
import pandas as pd
size = 900
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')

# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.svm import LinearSVR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_clf = make_pipeline(PolynomialFeatures(degree=4, include_bias=False), StandardScaler(), LinearSVR(random_state=42))
svm_clf.fit(X_train.reshape(-1, 1), y_train)

# %%
from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train, svm_clf.predict(X_train.reshape(-1, 1)))
mse_test = mean_squared_error(y_test, svm_clf.predict(X_test.reshape(-1, 1)))

print(mse_train)
print(mse_test)

# %%
from sklearn.svm import SVR

poly_kernel_svr_clf = make_pipeline(StandardScaler(), SVR(kernel="poly", degree=4))
poly_kernel_svr_clf.fit(X_train.reshape(-1, 1), y_train)

mse_train_svr = mean_squared_error(y_train, poly_kernel_svr_clf.predict(X_train.reshape(-1, 1)))
mse_test_svr = mean_squared_error(y_test, poly_kernel_svr_clf.predict(X_test.reshape(-1, 1)))

print(mse_train_svr)
print(mse_test_svr)

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid = {
    "C":[0.1, 1, 10],
    "coef0":[0.1, 1, 10]
}

model = SVR()

search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1)
search.fit(X.reshape(-1, 1), y)

print(search.best_params_)

best_params = search.best_params_
model_new = SVR(kernel='poly', degree=4, C=best_params['C'], coef0=best_params['coef0'])
model_new.fit(X_train.reshape(-1, 1), y_train)

mse_train_with_params = mean_squared_error(y_train, model_new.predict(X_train.reshape(-1, 1)))
mse_test_with_params = mean_squared_error(y_test, model_new.predict(X_test.reshape(-1, 1)))

# %%
arr = [mse_train, mse_test, mse_train_with_params, mse_test_with_params]
with open('reg_mse.pkl', 'wb') as file:
    pickle.dump(arr, file)

print(arr)


