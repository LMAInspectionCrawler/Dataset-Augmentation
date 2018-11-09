# Similar to cifar10_input
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 24		# crop images to 24x24

#Global constants
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

def read_images(filename_queue):
	""" Reads and parses images
	Args:
		filename_queue: A queue of strings with the filenames to read from.

	Returns:
		A record object with the following attributes:
				height: number of rows in the result
				width: number of cols in the result
				depth: number of color channels in the result (3)
				key: a scalar string Tensor describing the filename & record
				number for this example
				label: an int32 Tensor with the label in the range 0..9
				uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class ImageRecord(object):
		pass
	result = ImageRecord()
	# Dimensions of the original images
	label_bytes = 1
	result.height = 200
	result.width = 200
	result.depth = 3
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a
	# fixed number of bytes for each.
	record_bytes = label_bytes + image_bytes

	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)		# reads a record

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)

	# The first bytes represent the label, which we convert from uint8->int32.
	result.label = tf.cast(
		tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

	# The remaining bytes after the label represent the image, which we reshape
	# from [depth * height * width] to [depth, height, width].
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, [label_bytes],
		[label_bytes + image_bytes]),
		[result.depth, result.height, result.width])
	# Convert from [depth, height, width] to [height, width, depth].
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
									batch_size, shuffle):
	""" Construct a queued batch of images and labels
	Args:
		image: 3D Tensor of [height, width, 3] of type.float32
		label: 1D Tensor of type.int32
		min_queue_examples: int32, minimum number of sample to retain
			in the queue that provides of batches of examples
		batch_size: Number of images per batch

	Returns:
		images: 4D tensor of images [batch_size,, height, width, 3] size
		labels: 1D tensor of labels of [batch_size]
	"""
	# Grab batch_size images + labels from the a shuffled queue
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size)

	# Display images in the visualizer (TensorBoard?)
	# tf.summary.image('images', images)
	return images, tf.reshape(label_batch, [batch_size])

def augmented_inputs(data_dir, batch_size):
	""" Augments input to have more images for training
	Args:
		data_dir: Path to the image data directory
		batch_size: Number of images per batch
	
	Returns:
		images: 4D tensor of images [batch_size,, height, width, 3] size
		labels: 1D tensor of labels of [batch_size]
	"""

	# Get files
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
		for i in xrange(1, 6)]

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read
	filename_queue = tf.train.string_input_producer(filenames)

	with tf.name_scope('data_augmentation'):
		# Read images from files in the filename queue
		read_input = read_images(filename_queue)
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE
		width = IMAGE_SIZE

		# TODO: Don't crop, resize it
		cropped_image = tf.random_crop(reshaped_image, [height, width, 3])

		augmented_image = tf.image.random_flip_left_right(cropped_image)
		augmented_image = tf.image.random_brightness(augmented_image, max_delta = 63)
		augmented_image = tf.image.random_contrast(augmented_image, lower = 0.2, upper = 1.8)

		# Subtract off the mean and divide by the variance of the pixels
		float_image = tf.image.per_image_standardization(augmented_image)

		# Set the shapes of tensors
		float_image.set_shape([height, width, 3])
		read_input.label.set_shape([1])

		# Ensure that the random shuffling has good mixing properties
		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
								min_fraction_of_examples_in_queue)
		print ('Filling queue with %d images before starting to train. '
				'This will take a few minutes.' % min_queue_examples)

	return _generate_image_and_label_batch(float_image, read_input.label,
											min_queue_examples, batch_size,
											shuffle=True)



def inputs(eval_data, data_dir, batch_size):
	""" Load images for evaluation
	Args:
		eval_data: bool, indication if one should use the train or eval data set
		data_dir: path to the data directory
		batch_size: Number of images per batch

	Returns:
		images: 4D tensor of images [batch_size,, height, width, 3] size
		labels: 1D tensor of labels of [batch_size]
	"""
	if not eval_data:
		filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
			for i in xrange(1, 6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise Valuerror('Failed to find file: ' + f)

	with tf.name_scope('input'):
		# Create a queue that produces the filenames to read
		filename_queue = tf.train.string_input_producer(filenames)

		# Read images from files in the filename queue
		read_input = read_input(filename_queue)
		reshaped_image = tf.cast(read_input.uint8image, tf.float32)

		height = IMAGE_SIZE
		width = IMAGE_SIZE

		resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

		# Subtract off the mean and divide by the variance of the pixels
		float_image = tf.image.per_image_standardization(resized_image)

		# Set the shapes of tensors
		float_image.set_shape([height, width, 3])
		read_input.label.set_shape([1])

		# Ensure that the random shuffling has good mixing properties
		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(num_examples_per_epoch *
								min_fraction_of_examples_in_queue)

	return _generate_image_and_label_batch(float_image, read_input.label,
											min_queue_examples, batch_size,
											shuffle=False)















