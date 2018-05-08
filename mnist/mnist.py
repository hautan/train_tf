# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_size = 784
output_size = 10
hidden_layer_size = 50
hidden_layer_size2 = 200

tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights1 = tf.get_variable("weights1", [input_size, hidden_layer_size])
biases1 = tf.get_variable("biases1", [hidden_layer_size])

output1 = tf.nn.relu(tf.matmul(inputs, weights1) + biases1)

weights2 = tf.get_variable("weights2", [hidden_layer_size, hidden_layer_size])
biases2 = tf.get_variable("biases2", [hidden_layer_size])

output2 = tf.nn.sigmoid(tf.matmul(output1, weights2) + biases2)

weights3 = tf.get_variable("weights3", [hidden_layer_size, hidden_layer_size2])
biases3 = tf.get_variable("biases3", [hidden_layer_size2])

output3 = tf.nn.tanh(tf.matmul(output2, weights3) + biases3)

weights4 = tf.get_variable("weights4", [hidden_layer_size2, output_size])
biases4 = tf.get_variable("biases4", [output_size])


#output = tf.nn.relu(tf.matmul(output2, weights3) + biases3)
outputs = tf.matmul(output3, weights4) + biases4

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)
optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

sess = tf.InteractiveSession()

# Initialize the variables. Default initializer is Xavier.
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Batching
batch_size = 100

# Calculate the number of batches per epoch for the training set.
batches_number = mnist.train._num_examples // batch_size

# Basic early stopping. Set a miximum number of epochs.
max_epochs = 15

# Keep track of the validation loss of the previous epoch.
# If the validation loss becomes increasing, we want to trigger early stopping.
# We initially set it at some arbitrarily high number to make sure we don't trigger it
# at the first epoch
prev_validation_loss = 9999999.

import time
start_time = time.time()

# Create a loop for the epochs. Epoch_counter is a variable which automatically starts from 0.
for epoch_counter in range(max_epochs):
    
    # Keep track of the sum of batch losses in the epoch.
    curr_epoch_loss = 0.
    
    # Iterate over the batches in this epoch.
    for batch_counter in range(batches_number):
        
        # Input batch and target batch are assigned values from the train dataset, given a batch size
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        
        # Run the optimization step and get the mean loss for this batch.
        # Feed it with the inputs and the targets we just got from the train dataset
        _, batch_loss = sess.run([optimize, mean_loss], 
            feed_dict={inputs: input_batch, targets: target_batch})
        
        # Increment the sum of batch losses.
        curr_epoch_loss += batch_loss
    
    # So far curr_epoch_loss contained the sum of all batches inside the epoch
    # We want to find the average batch losses over the whole epoch
    # The average batch loss is a good proxy for the current epoch loss
    curr_epoch_loss /= batches_number
    
    # At the end of each epoch, get the validation loss and accuracy
    # Get the input batch and the target batch from the validation dataset
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    
    # Run without the optimization step (simply forward propagate)
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy], 
        feed_dict={inputs: input_batch, targets: target_batch})
    
    # Print statistics for the current epoch
    # Epoch counter + 1, because epoch_counter automatically starts from 0, instead of 1
    # We format the losses with 3 digits after the dot
    # We format the accuracy in percentages for easier interpretation
    print('Epoch '+str(epoch_counter+1)+
          '. Mean loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
    # Trigger early stopping if validation loss begins increasing.
    if validation_loss > prev_validation_loss:
        break
        
    # Store this epoch's validation loss to be used as previous validation loss in the next iteration.
    prev_validation_loss = validation_loss
    
input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})

test_accuracy_percent = test_accuracy[0] * 100.

# Print the test accuracy formatted in percentages
print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')

