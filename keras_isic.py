print("Importing libraries")

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	# Disables CPU AVX2 FMA warning since we can't use it

# Keras imports to build CNN model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam		# Adam optimizer is an advanced gradient descent algorithm

# TODO: Replace MNIST with ISIC images
#from mnist import MNIST
from load_isic import load_ISIC

# Modified From: https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/03C_Keras_API.ipynb

def print_dataset_information(data):
	print("Size of:")
	print("- Training-set:\t\t{}".format(data.num_train))
	print("- Validation-set:\t{}".format(data.num_val))
	print("- Test-set:\t\t{}".format(data.num_test))

def build_model(data):
	""" Builds sequential model """
	print("Building the model")
	# TODO: replace with a 3x(Conv + RELU + Pool) + 2x (FC)

	# Start construction of the Keras Sequential model.
	model = Sequential()

	# Add an input layer which is similar to a feed_dict in TensorFlow.
	# Note that the input-shape must be a tuple containing the image-size.
	#model.add(InputLayer(input_shape=(img_size_flat * num_channels,), name='input_layer'))
	#test = ((data.img_size_flat * data.num_channels),)
	#print "TESTING"
	#print("img_shape_full: " + str(data.img_shape_full))
	#print("img_size_flat: " + str(data.img_size_flat))

	#model.add(InputLayer(input_shape=((img_size_flat * num_channels),)))
	model.add(InputLayer(input_shape=(data.img_size_flat,)))

	# The input is a flattened array with 2352 elements,
	# but the convolutional layers expect images with shape (28, 28, 3)
	#test = Reshape(data.img_shape_full)
	#test = Reshape(data.img_size_flat, 1, data.img_size, data.img_size)
	#print str(type(test))
	model.add(Reshape(data.img_shape_full))
	#model.add(test)

	# First convolutional layer with ReLU-activation and max-pooling.
	model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
					activation='relu', name='layer_conv1'))
	model.add(MaxPooling2D(pool_size=2, strides=2))

	# Second convolutional layer with ReLU-activation and max-pooling.
	model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
					activation='relu', name='layer_conv2'))
	model.add(MaxPooling2D(pool_size=2, strides=2))

	# Flatten the 4-rank output of the convolutional layers
	# to 2-rank that can be input to a fully-connected / dense layer.
	model.add(Flatten())

	# First fully-connected / dense layer with ReLU-activation.
	model.add(Dense(128, activation='relu', name="layer_dense1"))

	# Last fully-connected / dense layer with softmax-activation
	# for use in classification.
	model.add(Dense(data.num_classes, activation='softmax', name="classify_layer"))


	model.summary()

	return model

def compile_model(model):
	""" Configures the model with the algorithms to use """
	print("Compiling the model")
	optimizer = Adam(lr=1e-3)
	model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])
	return model

def train(model, data):
	""" Trains the data with mini-batches """
	print("Begin training")
	callbacks = [
		# Interrupt training if `val_loss` stops improving for over 2 epochs
		tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
		# Write TensorBoard logs to `./logs` directory
		tf.keras.callbacks.TensorBoard(log_dir='./logs')
	]

	model.fit(
		x=data.x_train,
		y=data.y_train,
		epochs=10,
		batch_size=128,
		callbacks = callbacks)
	return model

def evaluate(model, data):
	""" Evaluates/tests the data by predicting and comparing to true labels """
	print("Begin evaluation")
	result = model.evaluate(
		x=data.x_test,
		y=data.y_test)

	print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

	return result

""" Helper functions """
def check_images(data):
	""" Plots the first 9 images with their labels to make sure the
	dataset is loaded correctly """

	# Get the first images from the test-set.
	images = data.x_test[0:9]

	# Get the true classes for those images.
	cls_true = data.y_test_cls[0:9]

	# Plot the images and labels using our helper-function
	plot_images(images=images, cls_true=cls_true)


def plot_images(images, cls_true, cls_pred=None):
	""" Plots 3x3 images with their true label """
	assert len(images) == len(cls_true) == 9
	
	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.
		ax.imshow(images[i].reshape(data.img_shape_full), cmap='binary')

		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		# Show the classes as the label on the x-axis.
		ax.set_xlabel(xlabel)
		
		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])
	
	# Ensure the plot is shown correctly with multiple plots
	# in a single Notebook cell.
	plt.show()

def plot_example_errors(cls_pred):
	""" Plots sample 3x3 images that were misclassified """

	# cls_pred is an array of the predicted class-number for
	# all images in the test-set.

	# Boolean array whether the predicted class is incorrect.
	incorrect = (cls_pred != data.y_test_cls)

	# Get the images from the test-set that have been
	# incorrectly classified.
	images = data.x_test[incorrect]
	
	# Get the predicted classes for those images.
	cls_pred = cls_pred[incorrect]

	# Get the true classes for those images.
	cls_true = data.y_test_cls[incorrect]
	
	# Plot the first 9 images.
	plot_images(images=images[0:9],
				cls_true=cls_true[0:9],
				cls_pred=cls_pred[0:9])

if __name__ == '__main__':
	#data = MNIST("data/MNIST/")
	data = load_ISIC("ISIC-images")

	print_dataset_information(data)
	check_images(data)

	model = build_model(data)	# TODO: refactor params
	model = compile_model(model)

	model = train(model, data)
	result = evaluate(model, data)