# Let's get you up and running with TensorFlow!

# But before we even get started, let's give you a sneak peek at what TensorFlow code looks like in the Python API, just so you have a sense of where we're headed.

# Here's a little Python program that makes up some data in three dimensions, and then fits a plane to it.

###############################################################################

# import libraries
import tensorflow as tf
import numpy as np

### Make 100 phony data point in NumPy ###

x_data = np.float32(np.random.rand(2, 100))
# 2 x 100 matrix is generated, with elements lying [0, 1)
# these are x and y coordinates of the 3D points

y_data = np.dot([0.800, 0.300], x_data) + 0.200
# the z coordinates of the 100 points calculated by  z = ax + by + c
# equation of a plane
# here -a = .8, -b = .3, c = .2

### Construct a linear model ###

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# tensorflow variables b and W are randomly selected
# their values will change during training
# final values should be b = .2 and W = [.8, .3]

y = tf.matmul(W, x_data) + b
# z coordinates of the guessed plane

### Minimize squared errors. ###

loss = tf.reduce_mean(tf.square(y - y_data))
# mean of squares of the difference between guessed and actual z coords.
optimizer = tf.train.GradientDescentOptimizer(0.03)
# set up grad desc optimizer w/ learning rate = 0.5
train = optimizer.minimize(loss)
# aka : W_new = W_old - 0.5 * gradient(loss)
#       b_new = b_old - 0.5 * gradient(loss)

### For initializing the variable. ###
init = tf.initialize_all_variables()

### Launch the graph ###
sess = tf.Session()
sess.run(init)

### Fit the plane ###
for step in xrange(0, 2001):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

