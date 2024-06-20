# coding: utf-8
# %%
from tensorflow import keras
import tensorflow as tf
import os

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# %%
X_train = X_train / 255
X_test = X_test / 255

# %%
import matplotlib.pyplot as plt
# plt.imshow(X_train[2137], cmap="binary")
# plt.axis('off')
# plt.show()


# %%
class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[2137]]


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, InputLayer
from tensorflow.keras import layers, models

model = models.Sequential()

model.add(layers.Flatten(input_shape=(28, 28)))

model.add(layers.Dense(300, activation='relu'))


model.add(layers.Dense(100, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

# %%
# model.summary()
# tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)

# %%
model.compile(loss="sparse_categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])

# %%
root_logdir = os.path.join(os.curdir, "image_logs")
root_logdir

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
run_logdir

# %%
tb_cb= tf.keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(X_train, y_train, epochs=20,
validation_split=0.1,
callbacks=[tb_cb])

# %%
# import numpy as np

# for i in range(10):
#     image_index = np.random.randint(len(X_test))
#     image = np.array([X_test[image_index]])
#     confidences = model.predict(image)
#     confidence = np.max(confidences[0])
#     prediction = np.argmax(confidences[0])
#     print("Prediction:", class_names[prediction])
#     print("Confidence:", confidence)
#     print("Truth:", class_names[y_test[image_index]])
#     plt.imshow(image[0], cmap="binary")
#     plt.axis('off')
#     plt.show()


# %%
model.save("fashion_clf.keras")

# %% [markdown]
# # HOUSE 1

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()

# %%
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# %%
model = models.Sequential()

norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
norm_layer.adapt(housing.target)

model.add(norm_layer)
model.add(layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]))
model.add(layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]))
model.add(layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]))
model.add(layers.Dense(1))


# %%
from keras.src.metrics import RootMeanSquaredError

model.compile(loss="mean_squared_error", optimizer="adam", metrics=[RootMeanSquaredError()])

early_cb = tf.keras.callbacks.EarlyStopping(patience=5, min_delta=0.01, verbose=1)

root_logdir = os.path.join(os.curdir, "housing_logs")
root_logdir
get_run_logdir()

tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[early_cb,tb_cb])

# %%
model.save("reg_housing_1.keras")

# %% [markdown]
# # HOUSE 2

# %%
model = models.Sequential()

norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
norm_layer.adapt(housing.target)

model.add(norm_layer)
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(1))

# %%
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[RootMeanSquaredError()])

tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[early_cb,tb_cb])

# %%
model.save("reg_housing_2.keras")

# %% [markdown]
# # HOUSE 3

# %%
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:], axis=None)
norm_layer.adapt(X_train)


model = tf.keras.models.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(70, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(120, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="adam", metrics=[RootMeanSquaredError()])
root_logdir = os.path.join(os.curdir, "housing_logs")
root_logdir
get_run_logdir()

tb_cb= keras.callbacks.TensorBoard(get_run_logdir())
history = model.fit(X_train, y_train, epochs=100,
validation_data=(X_valid, y_valid),
callbacks=[early_cb,tb_cb])

model.save("reg_housing_3.keras")



