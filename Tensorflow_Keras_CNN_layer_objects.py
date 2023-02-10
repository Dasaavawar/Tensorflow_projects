#Import libraries
import tensorflow as tf
from tensorflow.python.keras import layers, models
from keras.datasets import datasets
import matplotlib.pyplot as plt

#Load and split dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Lets look at one image
img_index = 1

plt.imshow(train_images[img_index], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[img_index][0]])
plt.show()

#CNN Architecture
model = models.sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#Lets have a loot at our model so far
model.summary()

#Adding dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#Training
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels))

#Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

#Check accuracy
print(test_acc)

#Data augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#Creates a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
 
 #Pick an image to transform
test_img = train_images[1]
img = image.img_to_array(test_img)
img = img.reshape((1,) + img.shape)

i=0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break
plt.show()

#Using a Pretrained Model

#Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

#Dataset
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

#Creates a function object that we can use to get labels
get_label_name = metadata.features['label'].int2str

#Display 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

#Data preprocessing

img_size2 = 160

#Returns an image that is reshapes to img_size2
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) -1
    image = tf.image.resize(image, (img_size2, img_size2))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

#Finally shuffle and batch the images
batch_size = 32
shuffle_buffer_size = 1000

train_batches = train.shuffle(shuffle_buffer_size).batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

for img, label in raw_train.take(2):
    print('Original shape:', img.shape)

for img, label in train.take(2):
    print('New shape:', img.shape)

#Picking a pretrained model
img_shape3 = (img_size2, img_size2, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape3,
include_top=False,
weights='imagenet')

#Specific information of the model
base_model.summary()

for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

#Static model
base_model.trainable = False

#Specific information of the model
base_model.summary()

#Adding our classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.summary()

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

#We can evaluate the model right now to see ho it does before training on our new images
initial_epochs = 3
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

#Now we can train it on our images
history = model.fit(
    train_batches,
    epochs = initial_epochs,
    validation_data = validation_batches
)

acc = history.history['accuracy']
print(acc)

model.save('dogs_vs_cats.h5')
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')






