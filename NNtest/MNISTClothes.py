import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# load dataset and load data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# shape of dataset
print(X_train_full.shape)
print(X_train_full.dtype)

# scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] / 255.0, y_train_full[5000:] / 255.0

# build the neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation=keras.activations.relu))
model.add(keras.layers.Dense(100, activation=keras.activations.relu))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

print(model.summary())
print(model.layers)
hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy]
              )

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

print(model.evaluate(X_test, y_test))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
print(y_test[:3])
