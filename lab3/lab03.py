import numpy as np
import pandas as pd

size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['x']], df[['y']], test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
model_pred_test = model.predict(X_test)
model_mse_test = mean_squared_error(y_test, model_pred_test)
model_pred_train = model.predict(X_train)
model_mse_train = mean_squared_error(y_train, model_pred_train)

neigh_3 = KNeighborsRegressor(n_neighbors=3)
neigh_3.fit(X_train, y_train)
neigh_3_pred = neigh_3.predict(X_test)
mse_neigh_3 = mean_squared_error(y_test, neigh_3_pred)
neigh_3_pred_train =  neigh_3.predict(X_train)
mse_neigh_3_train = mean_squared_error(y_train, neigh_3_pred_train)

neigh_5 = KNeighborsRegressor(n_neighbors=5)
neigh_5.fit(X_train, y_train)
neigh_5_pred = neigh_5.predict(X_test)
mse_neigh_5 = mean_squared_error(y_test, neigh_5_pred)
neigh_5_pred_train =  neigh_5.predict(X_train)
mse_neigh_5_train = mean_squared_error(y_train, neigh_5_pred_train)

polys_models = []
polys_preds_mse = []
polys_mse_trains = []
poly_features = []
for i in range(2, 6):
    poly = PolynomialFeatures(i)
    poly_features.append(poly)
    train_poly = poly.fit_transform(X_train)
    test_poly = poly.fit_transform(X_test)
    l_reg = LinearRegression()
    polys_models.append(l_reg)
    l_reg.fit(train_poly, y_train)
    polys_preds_mse.append(mean_squared_error(y_test, l_reg.predict(test_poly)))
    polys_mse_trains.append(mean_squared_error(y_train, l_reg.predict(train_poly)))

data = {}
data['train_mse'] = [model_mse_train, mse_neigh_3_train, mse_neigh_5_train] + [poly_mse_train for poly_mse_train in polys_mse_trains]
data['test_mse'] = [model_mse_test, mse_neigh_3, mse_neigh_5] + [poly_mse_test for poly_mse_test in polys_preds_mse]

new_df = pd.DataFrame(data, index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])
new_df.to_pickle('mse.pkl')
new_df.head(8)


import pickle

excercise_4 = []
excercise_4.append((model, None))
excercise_4.append((neigh_3, None))
excercise_4.append((neigh_5, None))
excercise_4.append((polys_models[0], poly_features[0]))
excercise_4.append((polys_models[1], poly_features[1]))
excercise_4.append((polys_models[2], poly_features[2]))
excercise_4.append((polys_models[3], poly_features[3]))

print(excercise_4)
with open('reg.pkl', 'wb') as f:
    pickle.dump(excercise_4, f)


#comment
