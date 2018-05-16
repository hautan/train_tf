# -*- coding: utf-8 -*-
 # We must always import the relevant libraries for our problem at hand. NumPy and TensorFlow are required for this example.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0, 10.0, 1000000)
y_data = (0.5*x_data) + 5

batch_size = 8

w = tf.Variable(0.5)
b = tf.Variable(0.5)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = w*xph + b

error = tf.reduce_sum(tf.square(yph-y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  batches = 10000
  for i in range(batches):
    rand_ind = np.random.randint(len(x_data),size=batch_size)
    print(rand_ind)
    feed = {xph:x_data[rand_ind], yph:y_data[rand_ind]}
    sess.run(train, feed_dict=feed)
  model_w, model_b = sess.run([w,b])


