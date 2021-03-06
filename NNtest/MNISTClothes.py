import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

# load dataset and load data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# shape of dataset
print(X_train_full.shape)
print(X_train_full.dtype)

# scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# build the neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation=keras.activations.relu))
model.add(keras.layers.Dense(100, activation=keras.activations.relu))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

# print(model.summary())
# print(model.layers)
hidden1 = model.layers[1]
# print(hidden1.name)
# print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
# print(weights)
# print(weights.shape)
# print(biases)
# print(biases.shape)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy]
              )

history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid))

# print(history)
# print(model.evaluate(X_test, y_test))


# print(y_proba.round(2))
# print(y_test[:3])

# test = tf.one_hot(y_test, 10)
# print(test)

from sklearn.model_selection import StratifiedKFold

skfolds = StratifiedKFold(n_splits=3)
ret = []
rety = []
for train_index, test_index in skfolds.split(X_train, y_train):
    # clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train[test_index]
    model.fit(X_train_folds, y_train_folds)
    y_pred = model.predict(X_test_fold)
    ret.extend(tf.argmax(y_pred, 1))
    rety.extend(y_test_fold)



print(confusion_matrix(ret, rety))

# Evaluating confusion matric
# res = tf.math.confusion_matrix(y_test, tf.argmax(y_proba, 1))

# Printing the result
#print('Confusion_matrix: ', res)

precision, recall, fscore, support = score(rety, ret)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba)
print(y_test[:3])
