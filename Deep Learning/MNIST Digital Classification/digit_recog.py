import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import cv2 # OpenCV library for image-processing
import os
import math
import datetime
import warnings
warnings.filterwarnings("ignore")

# Set the random seed. Setting the seed ensures reproducibility, meaning that running the same code multiple times with the same seed should produce the same results.
tf.random.set_seed(3)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

# Loading the MNIST data from keras.datasets
# Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test).
(x_train, y_train), (x_test, y_test) = mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Explore the Data -  It is a 28x28 matrix of integers (from 0 to 255). Each integer represents a color of a pixel.
print(pd.DataFrame(x_train[0]))

# Training data = 60,000 Images
# Test data = 10,000 Images
# Image dimension --> 28 x 28
# Grayscale Image --> 1 channel

# displaying the image
plt.imshow(x_train[25])
plt.show()

# print the corresponding label
print("y_train is: ", y_train[25])

# unique values in Y_train
print(np.unique(y_train))

# unique values in Y_test
print(np.unique(y_test))

# scaling the values. All the images have the same dimensions in this dataset, If not, we have to resize all the images to a common dimension [0,1]
x_train = x_train/255
x_test = x_test/255

# Building the Neural Network - setting up the layers of the Neural  Network
model = tf.keras.models.Sequential()

# Input layers.
model.add(tf.keras.layers.Flatten(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.002)
))

# Hidden layers.
model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.002)
))

# Output layers.
model.add(tf.keras.layers.Dense(
    units=10,
    activation=tf.keras.activations.softmax
))
print(model.summary())

log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# # compiling the Neural Network
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# # Train the model
training_history = model.fit(x_train, y_train, epochs=10,
                             validation_data=(x_test, y_test),
                            callbacks=[tensorboard_callback])

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='test set')
plt.legend()
plt.show()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='test set')
plt.legend()
plt.show()

# # # evaluate the trained model 
loss, accuracy = model.evaluate(x_test, y_test)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Save the model using the native Keras format
model.save('digits_recognition_model.keras')

# prediction on testing data
Y_pred_prob = model.predict(x_test)
# extract predictions with highest probabilites and detect what digits have been actually recognized.
Y_pred_value = np.argmax(Y_pred_prob, axis=1)
print(Y_pred_value)
# check prediction of first image
print(y_test[0])
print(Y_pred_value[0])

# more test examples and correspondent predictions 
numbers_to_display = 90
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15, 15))

for plot_index in range(numbers_to_display):    
    predicted_label = Y_pred_value[plot_index]
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == y_test[plot_index] else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(x_test[plot_index], cmap=color_map)
    plt.xlabel(predicted_label)

plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()