import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

#Download Keras datasets
fashion_mnist = keras.datasets.fashion_mnist

#Import train and test images and labels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Determine value shape
train_images.shape

#Lets take a look to one pixel
train_images[0, 23, 23] 

#Lets have a look to the first 10 training labels
train_labels[:10] 

#Label them
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Printing first sample
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
#plt.show()

#Data processing
test_images = test_images / 255.0
train_images = train_images / 255.0

#Building model
#input layer 1
#hidden layer 2
#output layer 3
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Test the model with 10 epochs
model.fit(train_images, train_labels, epochs=10)

#Grab information about the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

#Print that
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()



