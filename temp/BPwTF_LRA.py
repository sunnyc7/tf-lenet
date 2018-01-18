# This file modifies BPwTF.py so that we add low-rank approximation to the back
# propagation.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from global_definitions import *

## FUNCTIONS
# Sigmoid function
def sigma(x):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

# Sigmoid prime
def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

# Setup the model
a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# TODO: make this program's output stable across runs (the following attempt failed)
tf.set_random_seed(seed)

w_1 = tf.Variable(tf.truncated_normal([784, hidden_1]))
b_1 = tf.Variable(tf.truncated_normal([1, hidden_1]))
w_2 = tf.Variable(tf.truncated_normal([hidden_1, hidden_2]))
b_2 = tf.Variable(tf.truncated_normal([1, hidden_2]))
w_3 = tf.Variable(tf.truncated_normal([hidden_2, 10]))
b_3 = tf.Variable(tf.truncated_normal([1, 10]))

# The forward propagation
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)
z_3 = tf.add(tf.matmul(a_2, w_3), b_3)
a_3 = sigma(z_3)

# TODO: use softmax in the last layer

# Loss
diff = tf.subtract(a_3, y)

# Backward Propagation
d_z_3 = tf.multiply(diff, sigmaprime(z_3))
d_b_3 = d_z_3
d_w_3 = tf.matmul(tf.transpose(a_2), d_z_3)
d_a_2 = tf.matmul(d_z_3, tf.transpose(w_3))
d_z_2 = tf.multiply(d_a_2, sigmaprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

##################################################
#    low-rank approximation on d_w_2             #
##################################################

# perform svd of d_w_2
s, u, v = tf.svd(d_w_2)
# truncate u, s, v
u_hat = u[:, :k]
s_hat = s[:k]
v_hat = v[:, :k]
# compute low-rank approximation of d_w_2
d_w_2_hat = tf.matmul(u_hat, tf.matmul(tf.diag(s_hat), tf.transpose(v_hat)))
# replace the original gradient matrix
d_w_2 = d_w_2_hat
##################################################

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

# Updating the network
eta = tf.constant(0.5)  # learning rate
step = [
    tf.assign(w_1,tf.subtract(w_1, tf.multiply(eta, d_w_1))),
    tf.assign(b_1,tf.subtract(b_1, tf.multiply(eta,tf.reduce_mean(d_b_1, axis=[0])))),
    tf.assign(w_2,tf.subtract(w_2, tf.multiply(eta, d_w_2))),
    tf.assign(b_2,tf.subtract(b_2, tf.multiply(eta,tf.reduce_mean(d_b_2, axis=[0])))),
    tf.assign(w_3,tf.subtract(w_3, tf.multiply(eta, d_w_3))),
    tf.assign(b_3,tf.subtract(b_3, tf.multiply(eta,tf.reduce_mean(d_b_3, axis=[0]))))
]

# Running and testing the training process
acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))

# number of correct answers
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict={a_0: batch_xs,y: batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict={a_0: mnist.test.images[:1000],
            y: mnist.test.labels[:1000]})
        print("iter = %-10d" % i, "correct prediction = %d/1000" % res)
