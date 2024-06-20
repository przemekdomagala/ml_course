# %% [markdown]
# # 2.1, 2.2

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# %%
import numpy as np
from scipy.stats import reciprocal

param_distribs = {
"model__n_hidden": [0,1,2,3],
"model__n_neurons": np.arange(1, 101),
"model__learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
"model__optimizer": ['adam', 'sgd', 'nesterov']
}

# %%
def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'nesterov':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
                
    model.compile(loss='mse', optimizer=optimizer)
    return model 

# %%
import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])

# %%
from sklearn.model_selection import RandomizedSearchCV

rnd_search_cv = RandomizedSearchCV(keras_reg,
param_distribs,
n_iter=5,
cv=3,
verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid,
y_valid), verbose=0)

# %%
import pickle

dict_to_file = {
    'model__optimizer': rnd_search_cv.best_params_['model__optimizer'],
    'model__n_neurons': rnd_search_cv.best_params_['model__n_neurons'],
    'model__n_hidden': rnd_search_cv.best_params_['model__n_hidden'],
    'model__learning_rate': rnd_search_cv.best_params_['model__learning_rate']
}

with open('rnd_search_params.pkl', 'wb') as params_file:
    pickle.dump(rnd_search_cv.best_params_, params_file)

with open('rnd_search_scikeras.pkl', 'wb') as rnd_file:
    pickle.dump(rnd_search_cv, rnd_file)

# %% [markdown]
# # 2.3

# %%
import keras_tuner as kt

def build_model_kt(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=3, default=2)
    n_neurons = hp.Int("n_neurons", min_value=1, max_value=100)
    learning_rate = hp.Float("learning_rate", min_value=3e-4, max_value=3e-2, sampling="log")
    optimizer = hp.Choice("optimizer", ["sgd", "adam", "nesterov"])
    if optimizer == "sgd":
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nesterov":
        optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    elif optimizer == "adam":
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer=optimizer_instance, metrics=["mse"])    
    
    return model

# %%
random_search_tuner = kt.RandomSearch(
build_model_kt, objective="val_mse", max_trials=10, overwrite=True,
directory="my_california_housing", project_name="my_rnd_search", seed=42)

# %%
import os

root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)

# %%
random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[es, tb])

# %%
with open('kt_search_params.pkl', 'wb') as f:
     pickle.dump(random_search_tuner.get_best_hyperparameters()[0].values, f)
     
best_model = random_search_tuner.get_best_models()[0]
best_model.save('kt_best_model.keras')
