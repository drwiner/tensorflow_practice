"""
Replacing a simple function with tensorflow operation

"""

import tensorflow as tf


class tf_helper(object):
	x = tf.constant(2.0)
	tf_z = tf.placeholder(tf.float32)
	sess = tf.Session()
	comp = tf.add(x, tf_z)


	def tf_add_x(self, z):
		return self.sess.run(self.comp, feed_dict={self.tf_z:z})


if __name__ == "__main__":
	tf_obj = tf_helper()
	v = tf_obj.tf_add_x(12)
	print(v)