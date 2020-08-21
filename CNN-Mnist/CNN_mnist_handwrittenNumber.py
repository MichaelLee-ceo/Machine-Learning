import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
import mnist_displayNumber as md
from tensorflow.keras.callbacks import TensorBoard

category = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train2 = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test2 = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train2 /= 255
x_test2 /= 255

y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Conv2D(filters=40, kernel_size=(2, 2), padding="same", activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(units=category, activation=tf.nn.softmax))
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.003), metrics=['accuracy'])

# tensorBoard = TensorBoard(log_dir='logs')

result = model.fit(x_train2, y_train2, validation_split=0.2, batch_size=100, epochs=20, verbose=2)

train_scores = model.evaluate(x_train2, y_train2)
print('Train accuracy:', train_scores[1])
test_scores = model.evaluate(x_test2, y_test2)
print('Test accuracy:', test_scores[1])

md.show_train_history(result, 'accuracy', 'val_accuracy')
model.save('C:\\Users\\USER\\Desktop\\Python TensorFlow\\model\\cnn.h5')