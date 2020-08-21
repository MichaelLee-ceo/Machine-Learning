import tensorflow as tf
import mnist_displayNumber as md

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # show training set number
# md.printMatrix(x_train[0])
# print('y_train[0]:', y_train[0])
# md.displayMatrix(x_train, 0, y_train[0])

dim = x_train.shape[1] * x_train.shape[2]
x_train2 = x_train.reshape(x_train.shape[0], dim).astype('float32')
x_test2 = x_test.reshape(x_test.shape[0], dim).astype('float32')

x_train2 /= 255
x_test2 /= 255

y_train2 = tf.keras.utils.to_categorical(y_train, 10)
y_test2= tf.keras.utils.to_categorical(y_test, 10)

learning_rate = 0.001
decay_steps = 10000
decay_rate = 1

decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,     # 初始学习率
    decay_steps,       # 衰减周期
    decay_rate,        # 衰减率系数
    staircase=False,   # 定义是否是阶梯型衰减，还是连续衰减，默认是 False
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_dim=dim, kernel_regularizer=tf.keras.regularizers.l2(l=0.003)))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=decay_lr, beta_1=0.9, beta_2=0.99), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()
# input()
result = model.fit(x_train2, y_train2, validation_split=0.2, batch_size=512, epochs=100, verbose=2)

# train_scores = model.evaluate(x_train2, y_train2)
# print('Train accuracy:', train_scores[1])
test_scores = model.evaluate(x_test2, y_test2)
print('Test accuracy:', test_scores[1])

md.show_train_history(result, 'loss', 'val_loss')

# predict = model.predict_classes(x_test2)
# md.plot_images_labels_prediction(x_test, y_test, predict, idx=0, num=10)
# predict = model.predict_classes(x_test[:10])
# print("Predict classes:", predict)
# print("y_test:", y_test[:10])