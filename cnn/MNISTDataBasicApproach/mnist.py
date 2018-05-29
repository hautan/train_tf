# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
type(mnist)
mnist.train.images
mnist.train.num_examples
mnist.test.num_examples
mnist.validation.num_examples
mnist.train.images[1].shape
plt.imshow(mnist.train.images[1].reshape(28,28))
plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
mnist.train.images[1].max()
plt.imshow(mnist.train.images[1].reshape(784,1))
plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)


input_size = 784
output_size = 10

tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=[None, input_size])
y_true = tf.placeholder(tf.float32,[None, output_size])

W = tf.Variable(tf.zeros([input_size, output_size]))
b = tf.Variable(tf.zeros([output_size]))
# Create the Graph
y = tf.matmul(x,W) + b 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience
    
    for step in range(1000):        
        batch_x , batch_y = mnist.train.next_batch(100)        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    # Test the Train Model
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))    
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))