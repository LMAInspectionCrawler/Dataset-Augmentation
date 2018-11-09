import numpy as np
import json
from glob import glob 	# To read files
from PIL import Image

CLASSES = 2		# benign/malignant
CHANNELS = 3
SUBSET_DISTRO = [0.8, 0.0, 0.2]		# Percentage to cut into training, validation, and testing subsets respectively
WIDTH = HEIGHT = 28

def load_ISIC(path):
	""" Returns the ISIC dataset """
	data = Dataset()
	images, labels = read_files(path)
	# TODO: reshape images

	data.img_size = WIDTH		# Assuming width = height
	data.img_size_flat = HEIGHT * WIDTH 	# When image is flattened into 1D array
	data.img_shape = HEIGHT, WIDTH
	data.img_shape_full = HEIGHT, WIDTH, CHANNELS
	data.num_classes = CLASSES
	data.num_channels = CHANNELS

	# Using desired training/validation/testing ratios to get splice the images into subsets
	train_stop_point = int(len(images) * SUBSET_DISTRO[0])
	val_stop_point = int(len(images) * SUBSET_DISTRO[1]) + train_stop_point

	data.x_train = images[0:train_stop_point]
	data.y_train = images[0:train_stop_point]		# Gets the one-hot classes (ex. 0.1, 0.0, 0.9)
	data.y_train_cls = getTrueClasses(data.y_train)

	data.x_val = images[train_stop_point:val_stop_point]
	data.y_val = images[train_stop_point:val_stop_point]
	data.y_val_cls = getTrueClasses(data.y_val)

	data.x_test = images[val_stop_point:]
	data.y_test = images[val_stop_point:]
	data.y_test_cls = getTrueClasses(data.y_test)

	data.num_train = len(data.x_train)
	data.num_val = len(data.x_val)
	data.num_test = len(data.x_test)

	return data

def read_files(path):
	"""
	Returns a tuple of numpy arrays.
	The first is the ISIC jpg images, the second is the classification benign => 0, malignant => 1
	"""

	# Read file lists
	image_filename_list = glob(path + "/*.jpg")
	metadata_filename_list = glob(path + "/*.json")

	# Read images and add to numpy array
	image_arr = []
	for i in range(0, len(image_filename_list)):
		#image = open(image_filename_list[i])
		image = Image.open(image_filename_list[i])	# Need to keep the file open to get information
		resizedImage = image.resize((WIDTH, HEIGHT))
		imageArr = np.array(resizedImage)
		image_1D = imageArr.ravel()
		image_arr.append(image_1D)

	image_arr = np.array(image_arr)

	# Parse json metadata, get label, convert label to one-hot, and add to numpy array
	label_arr = np.empty([len(image_arr), CLASSES])
	for i in range(0, len(image_arr)):
		with open(metadata_filename_list[i]) as f:
			json_data = json.load(f)
		label = json_data['meta']['clinical']['benign_malignant']
		one_hot_label = convert_label_to_one_hot(label)
		label_arr[i] = one_hot_label

	return image_arr, label_arr

def getTrueClasses(one_hot_classes):
	trueClasses = np.empty(len(one_hot_classes))

	for each_class_set_index in range(0, len(one_hot_classes)):
		each_class_set = one_hot_classes[each_class_set_index]
		trueClasses[each_class_set_index] = np.argmax(each_class_set)

	return trueClasses

def convert_label_to_one_hot(label):
	one_hot_label = np.array([0, 0])
	if label == 'benign':
		one_hot_label = np.array([1, 0])
	elif label == 'malignant':
		one_hot_label = np.array([0, 1])
	else: 
		print "Error: Unknown label. Label should be benign or malignant"
	return one_hot_label

class Dataset:
	pass