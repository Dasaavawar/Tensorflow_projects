from pickletools import optimize
import tensorflow as tf
import numpy as np

train_in = np.random.randint(0,90, (100000,1))
train_out = train_in.flatten()

test_in = np.random.randint(0,90, (10000,1))
test_out = test_in.flatten()

model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(1,1)),
    tf.keras.layers.Dense(1024, activation='sigmoid'),
    tf.keras.layers.Dense(512, activation='sigmoid'),
    tf.keras.layers.Dense(256, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(91, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_in, train_out, epochs=10)

#print(model.evaluate(test_in, test_out))

age = 69
prediction = model.predict([age])
print('You are: ', np.argmax(prediction[0]))