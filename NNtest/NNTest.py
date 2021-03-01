import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# load dataset and load data
import pandas
df = pandas.read_csv(r'./heart.csv')
print(df.shape)
df

df_x = df.drop(['target'], axis=1)
df_y = df['target']

df_x = df_x.to_numpy()
df_y = df_y.to_numpy()

X_train_full, X_test,y_train_full, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
# shape of dataset
print(X_train_full.shape)
print(X_train_full.dtype)

# scaling the features
X_valid, X_train = X_train_full[:203], X_train_full[203:]
y_valid, y_train = y_train_full[:203], y_train_full[203:]

y_train_5 = (y_train == 5) * 0.99
y_test_5 = (y_test == 5) * 0.99
y_valid_5 = (y_valid == 5) * 0.99

# build the neural network
model = keras.models.Sequential()
model.add(keras.layers.Dense(14, activation=keras.activations.relu))
model.add(keras.layers.Dense(300, activation=keras.activations.relu))
model.add(keras.layers.Dense(100, activation=keras.activations.relu))
model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.Accuracy(),
                       keras.metrics.Precision(),
                       keras.metrics.Recall()]
              )

y_train_full_cont = y_train_full * 0.99
y_test_cont = y_test * 0.99

history = model.fit(X_train_full, y_train_full_cont, epochs=29, validation_data=(X_test, y_test_cont))

from sklearn.model_selection import StratifiedKFold, train_test_split

skfolds = StratifiedKFold(n_splits=3)
ret = []
rety = []
for train_index, test_index in skfolds.split(X_train_full, y_train_full):
    # clone_clf = clone(sgd_clf)
    X_train_folds = X_train_full[train_index]
    y_train_folds = y_train_full[train_index] * 0.99
    X_test_fold = X_train_full[test_index]
    y_test_fold = y_train_full[test_index] * 0.99
    model.fit(X_train_folds, y_train_folds)
    y_pred = model.predict(X_test_fold)
    ret.extend(y_pred > 0.5)
    rety.extend(y_test_fold == 0.99)

print(confusion_matrix(ret, rety))

# Evaluating confusion matric
# res = tf.math.confusion_matrix(y_test, tf.argmax(y_proba, 1))

# Printing the result
# print('Confusion_matrix: ', res)

precision, recall, fscore, support = score(rety, ret)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba)
print(y_test[:3])

print(history)
print(model.evaluate(X_test, y_test_5))

X_new = X_test[:3]
y_proba = model.predict(X_new) > 0.5
print(y_proba)
print(y_test[:3])
# print(y_proba.round(2))
# print(y_test[:3])

# test = tf.one_hot(y_test, 10)
# print(test)

# Evaluating confusion matric
#res = tf.math.confusion_matrix(y_test_5, y_proba)

# Printing the result
#rint('Confusion_matrix: ', res)
