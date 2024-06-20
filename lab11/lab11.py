#!/usr/bin/env python
# coding: utf-8

# %%
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras

[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
"tf_flowers",
split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
as_supervised=True,
with_info=True)

# %%
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

# %%
import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# index = 0
# sample_images = train_set_raw.take(9)
# for image, label in sample_images:
#     index += 1
#     plt.subplot(3, 3, index)
#     plt.imshow(image)
#     plt.title("Class: {}".format(class_names[label]))
#     plt.axis("off")
# plt.show(block=False)

# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label

# %%
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# %%
# plt.figure(figsize=(8, 8))
# sample_batch = train_set.take(1)
# print(sample_batch)
# for X_batch, y_batch in sample_batch:
#     for index in range(12):
#         plt.subplot(3, 4, index + 1)
#         plt.imshow(X_batch[index]/255.0)
#         plt.title("Class: {}".format(class_names[y_batch[index]]))
#         plt.axis("off")
# plt.show()

# %%
model = keras.models.Sequential([
    keras.layers.Rescaling(scale=1./255),
    keras.layers.Conv2D(filters=32, kernel_size=7, activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(n_classes, activation='softmax'),
])

# %%
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# %%
history = model.fit(train_set, epochs=10, validation_data=(valid_set))

# %%
score_train = model.evaluate(train_set)[1]
score_valid = model.evaluate(valid_set)[1]
score_test = model.evaluate(test_set)[1]
print(score_train + score_valid + score_test)

# %%
import pickle

simple_cnn_acc = (score_train, score_valid, score_test)

with open('simple_cnn_acc.pkl', 'wb') as cnn_acc_file:
    pickle.dump(simple_cnn_acc, cnn_acc_file)

model.save('simple_cnn_flowers.keras')

# %% [markdown]
# # Uczenie Transferowe

# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# %%
# plt.figure(figsize=(8, 8))
# sample_batch = train_set.take(1).repeat()
# for X_batch, y_batch in sample_batch:
#     for index in range(12):
#         plt.subplot(3, 4, index + 1)
#         plt.imshow(X_batch[index] / 2 + 0.5)
#         plt.title("Class: {}".format(class_names[y_batch[index]]))
#         plt.axis("off")
# plt.show()

# %%
base_model = tf.keras.applications.xception.Xception(
weights="imagenet",
include_top=False)

# %%
#tf.keras.utils.plot_model(base_model)

# %%
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)


# %%
from tensorflow.keras.optimizers import Adam

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
history = model.fit(train_set, validation_data=valid_set, epochs=5)

# %%
for layer in base_model.layers:
    layer.trainable = False

history = model.fit(train_set, validation_data=valid_set, epochs=4)

# %%
score_train = model.evaluate(train_set)[1]
score_valid = model.evaluate(valid_set)[1]
score_test = model.evaluate(test_set)[1]
print(score_train + score_valid + score_test)

# %%
xception_acc = (score_train, score_valid, score_test)

with open('xception_acc.pkl', 'wb') as xception_acc_file:
    pickle.dump(xception_acc, xception_acc_file)

model.save('xception_flowers.keras')
