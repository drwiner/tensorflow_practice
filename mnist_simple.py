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
learning_rate = 0.5

# Number of epochs
num_epochs = 1000

"""
Random values
"""

# Random values with weight matrix dimensions, with standard deviation 0.1
random_matrix = tf.random_normal([num_images, num_labels], stddev=0.1)

# Random values for logit vector
random_logit = tf.random_normal([num_labels], stddev=0.1)

"""
Variables
"""
# Weight matrix (num_images x num_labels)
weight_matrix = tf.Variable(random_matrix)

# Y intercept
b = tf.Variable(random_logit)

"""
Placeholders
"""

# Image placeholder (batch x images)
image = tf.placeholder(tf.float32, [batch_size, num_images])

# Output matrix from batch (batches x labels) - each row a one-hot vector
batch_output = tf.placeholder(tf.float32, [batch_size, num_labels])

"""
Softmax to produce logits
"""
# Image (batches x images) x weights (images x label) = batch (batches x labels)
# A batch-sized array of logits
probs = tf.nn.softmax(tf.matmul(image, weight_matrix) + b)

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

# Initialize global variables
global_init = tf.global_variables_initializer()

# Run session
session.run(global_init)

######################################################################
""" Training """
######################################################################


for i in range(num_epochs):
	img, batch = mnist.train.next_batch(batch_size)
	acc, _ = session.run([acc_function, optimizer], feed_dict={image: img, batch_output: batch})
	print("acc:\t{}".format(acc))
	
sum_acc = 0
for i in range(num_epochs):
	img, batch = mnist.test.next_batch(batch_size)
	sum_acc += session.run(acc_function, feed_dict={image: img, batch_output: batch})
	
print("Test Accuracy:\t%r" % str(sum_acc / float(num_epochs)))


