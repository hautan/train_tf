#https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from helper.cifar_input_data_helper import CifarInputDataHelper
from model_builder import CNNModel

tf.logging.set_verbosity(tf.logging.INFO)

ch = CifarInputDataHelper()
ch.set_up_images()

model = CNNModel(ch)
model.train_model()
model.eval_model()
print(model.eval_results)

model.predict([])
print(model.predictions)

#def cnn_model_fn(features, labels, mode):
#  """Model function for CNN."""
#  # Input Layer
#  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
#  # CIFAR images are 32x32 pixels, and have one color channel
#  input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
#
#  # Convolutional Layer #1
#  # Computes 32 features using a 5x5 filter with ReLU activation.
#  # Padding is added to preserve width and height.
#  # Input Tensor Shape: [batch_size, 32, 32, 1]
#  # Output Tensor Shape: [batch_size, 32, 32, 32]
#  conv1 = tf.layers.conv2d(
#      inputs=input_layer,
#      filters=32,
#      kernel_size=[5, 5],
#      padding="same",
#      activation=tf.nn.relu)
#
#  # Pooling Layer #1
#  # First max pooling layer with a 2x2 filter and stride of 2
#  # Input Tensor Shape: [batch_size, 32, 32, 32]
#  # Output Tensor Shape: [batch_size, 16, 16, 32]
#  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
#  # Convolutional Layer #2
#  # Computes 64 features using a 5x5 filter.
#  # Padding is added to preserve width and height.
#  # Input Tensor Shape: [batch_size, 16, 16, 32]
#  # Output Tensor Shape: [batch_size, 16, 16, 64]
#  conv2 = tf.layers.conv2d(
#      inputs=pool1,
#      filters=64,
#      kernel_size=[5, 5],
#      padding="same",
#      activation=tf.nn.relu)
#
#  # Pooling Layer #2
#  # Second max pooling layer with a 2x2 filter and stride of 2
#  # Input Tensor Shape: [batch_size, 16, 16, 64]
#  # Output Tensor Shape: [batch_size, 8, 8, 64]
#  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
#  # Flatten tensor into a batch of vectors
#  # Input Tensor Shape: [batch_size, 8, 78 64]
#  # Output Tensor Shape: [batch_size, 8 * 8 * 64]
#  pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
#
#  # Dense Layer
#  # Densely connected layer with 1024 neurons
#  # Input Tensor Shape: [batch_size, 8 * 8 * 64]
#  # Output Tensor Shape: [batch_size, 1024]
#  dense = tf.layers.dense(inputs=pool2_flat, units=ct.DROP_OUT_RATE, activation=tf.nn.relu)
#
#  # Add dropout operation; 0.6 probability that element will be kept
#  dropout = tf.layers.dropout(inputs=dense, rate=ct.DROP_OUT_RATE, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#  # Logits layer
#  # Input Tensor Shape: [batch_size, 1024]
#  # Output Tensor Shape: [batch_size, 10]
#  logits = tf.layers.dense(inputs=dropout, units=ct.OUTPUT)
#
#  predictions = {
#      # Generate predictions (for PREDICT and EVAL mode)
#      "classes": tf.argmax(input=logits, axis=1),
#      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#      # `logging_hook`.
#      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#  }
#  if mode == tf.estimator.ModeKeys.PREDICT:
#    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#  # Calculate Loss (for both TRAIN and EVAL modes)
##  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=ct.OUTPUT)
#  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
#
#  # Configure the Training Op (for TRAIN mode)
#  if mode == tf.estimator.ModeKeys.TRAIN:
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=ct.LEARNING_RATE)
#    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
#    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#  # Add evaluation metrics (for EVAL mode)
#  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
#  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
#train_data = np.asarray(ch.training_images, dtype=np.float32) # Returns np.array
#train_labels = np.asarray(ch.training_labels, dtype=np.int32)
#eval_data = np.asarray(ch.test_images, dtype=np.float32) # Returns np.array
#eval_labels = np.asarray(ch.test_labels, dtype=np.int32)
#
## Create the Estimator
#cifar_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=ct.MODEL_PATH)
#
## Set up logging for predictions
#tensors_to_log = {"probabilities": "softmax_tensor"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=ct.LOGGING_ITER_COUNT)
#
## Train the model
#train_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": train_data},
#    y=train_labels,
#    batch_size=ct.TRAIN_BATCH_SIZE,
#    num_epochs=None,
#    shuffle=True)
#
##cifar_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
#cifar_classifier.train(input_fn=train_input_fn, steps=ct.TRAIN_STEPS)
#
## Evaluate the model and print results
#eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": eval_data},
#    y=eval_labels,
#    num_epochs=1,
#    shuffle=False)
#eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
#print(eval_results)
#
#pred_input_func = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, shuffle=False)
#predictions = []
#for predict in cifar_classifier.predict(input_fn=pred_input_func):
#    predictions.append(predict)
#
#
#predictions

#import matplotlib.pyplot as plt
#%matplotlib inline
#import numpy as np
#X = eval_data
#X.shape
#
#plt.imshow(X[6])
#
#pred =  np.asarray([ p['classes'] for p in predictions ], dtype=np.int32)
#concatenated_result = np.vstack([pred, eval_labels])
#print(concatenated_result[0,1])
#
#failed_predictions = []
#len_res = len(concatenated_result[0])
#for i in range(len_res):
#    if concatenated_result[0][i] != concatenated_result[1][i]:
#        failed_predictions.append(concatenated_result[0][i])
