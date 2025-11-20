import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
x_test_flattened = x_test.reshape(len(x_test), 28 * 28)

model=keras.Sequential([
    layers.Dense(128, activation='sigmoid', input_shape=(784,)),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_flattened,y_train,epochs=5)
model.evaluate(x_test_flattened,y_test)
plt.matshow(x_test[0])
plt.show()
y_predicted=model.predict(x_test_flattened)
print(y_predicted[0])
print(np.argmax(y_predicted[0]))
y_predicted_labels=([np.argmax(i) for i in y_predicted])
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
print(classification_report(y_test,y_predicted_labels))

