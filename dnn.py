import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
dev=tf.config.list_physical_devices()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8)
model.evaluate(x_test,  y_test, verbose=2)
print(model.predict_classes(x_test[:10]))
tf.keras.models.save_model(model,'./modelo.h5')