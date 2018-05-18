# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.reset_default_graph()

x_data = np.linspace(0.0, 10.0, 1000000)
y_data = (1.7*x_data) + 3

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

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  batches = 10000
  for i in range(batches):
    rand_ind = np.random.randint(len(x_data),size=batch_size)
    print(rand_ind)
    feed = {xph:x_data[rand_ind], yph:y_data[rand_ind]}
    sess.run(train, feed_dict=feed)
  model_w, model_b = sess.run([w,b])
  saver.save(sess,'models/6_model.ckpt')
  

x_test = np.linspace(-1,11,10)
y_pred_plot = model_w*x_test + model_b

plt.plot(x_test,y_pred_plot,'r')
plt.plot(x_data,y_data,'*')
print("slope: ", model_w, "intercept", model_b)

#with tf.Session() as sess:    
#    # Restore the model
#    saver.restore(sess,'models/my_first_model.ckpt')
#    # Fetch Back Results
#    restored_w , restored_b = sess.run([w,b])
#    
#x_test = np.linspace(-1,11,10)
#y_pred_plot = restored_w*x_test + restored_b
#
#plt.plot(x_test,y_pred_plot,'r')
#plt.plot(x_data,y_data,'*')