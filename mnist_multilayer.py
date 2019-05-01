import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST data-set, using one-hot vectors
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Problem dimensions
"""

# Size of batch
batch_size = 100

# Number of images to train
num_images = 784

# Number of labels
num_labels = 10

# Learning rate
learning_rate = 0.05

# Number of epochs
num_epochs = 1000

"""
Random values
"""

# random for U
r_U = tf.random_normal([num_images, num_images], stddev=0.1)

# random for V
r_V = tf.random_normal([num_images, num_labels], stddev=0.1)

# random for bU
r_bU = tf.random_normal([num_images], stddev=0.1)

# random for bV
r_bV = tf.random_normal([num_labels], stddev=0.1)

"""
Variables
"""

# Layer 1 (images to hidden layer)
w_U = tf.Variable(r_U)
b_U = tf.Variable(r_bU)

# Layer 2 (hidden layer to label)
w_V = tf.Variable(r_V)
b_V = tf.Variable(r_bV)

"""
Input Output
"""

# Image placeholder (batch x images)
image = tf.placeholder(tf.float32, [batch_size, num_images])

# Output matrix from batch (batches x labels) - each row a one-hot vector
batch_output = tf.placeholder(tf.float32, [batch_size, num_labels])

"""
Neural Network Layers
"""

# L1_Output
hidden_layer = tf.matmul(image, w_U) + b_U

# \rho(xU + bU)
hidden_layer = tf.nn.relu(hidden_layer)

# L2 output (\rho(xU + bU) * V + bV)
probs = tf.matmul(hidden_layer, w_V) + b_V

# softmax(L2 output)
probs = tf.nn.softmax(probs)

# Replace each value with log(value)
log_probs = tf.log(probs)

# Element by element matrix multiplication (batch_output x logits)
one_hot_probs = batch_output * log_probs

# Parallelization and sum by rows, each row is batch output
per_image_sum = -tf.reduce_sum(one_hot_probs, reduction_indices=[1])

# Cross entropy
x_entropy = tf.reduce_mean(per_image_sum)

"""
Training parameters
"""

# Backward pass by gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(x_entropy)

# Batched-sized array of maximum logit positions
max_logit = tf.argmax(probs, 1)

# Batch-sized array of maximum one-hot vector positions
max_batch = tf.argmax(batch_output, 1)

# Calculate the number of answers (in batch output) match the logits
num_correct = tf.equal(max_logit, max_batch)

# Calculate accuracy
acc_function = tf.reduce_mean(tf.cast(num_correct, tf.float32))

"""
Session
"""

# Create session
session = tf.Session()

save_obj = tf.train.Saver()


def run(checkpoint=None, checkpoint_name=None, new_checkpoint_name=None):
	if checkpoint_name is None:
		checkpoint_name = "myLatest.ckpt"
	if new_checkpoint_name is None:
		new_checkpoint_name = "myLatest.ckpt"

	if checkpoint is not None and checkpoint:
		# restore instead of instantiating variables
		save_obj.restore(session, checkpoint_name)
	else:
		# Initialize global variables
		global_init = tf.global_variables_initializer()
		
		# Run session
		session.run(global_init)

	
	######################################################################
	""" Training """
	######################################################################
	
	
	epochs_per_save = 100
	
	for i in range(num_epochs):
		img, batch = mnist.train.next_batch(batch_size)
		acc, _ = session.run([acc_function, optimizer], feed_dict={image: img, batch_output: batch})
		if i % epochs_per_save == 0:
			print("acc:\t{}".format(acc))
			save_obj.save(session, new_checkpoint_name)
	
	sum_acc = 0
	for i in range(num_epochs):
		img, batch = mnist.test.next_batch(batch_size)
		sum_acc += session.run(acc_function, feed_dict={image: img, batch_output: batch})
	
	print("Test Accuracy:\t%r" % str(sum_acc / float(num_epochs)))


if __name__ == "__main__":
	
	# What actions should occur
	previous_checkpoint_name = "tmp/myLatest.ckpt"
	
	# Do start from checkpoint, or None for startover
	do_checkpoint = None
	
	# Run code to train and test
	run(checkpoint=do_checkpoint, checkpoint_name=previous_checkpoint_name, new_checkpoint_name=previous_checkpoint_name)
