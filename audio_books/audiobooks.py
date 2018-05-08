import numpy as np
from audiobook_data_reader import Audiobooks_Data_Reader

import tensorflow as tf
 
# Input size depends on the number of input variables. We have 10 of them
input_size = 10
# Output size is 2 as we one-hot encoded the targets.
output_size = 2
# Choose a hidden_layer_size
hidden_layer_size = 100

# Reset the default graph so you can fiddle with the hyperparameters and then rerun the code.
tf.reset_default_graph()

# Create the placeholders
inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

# Outline the model. We will create a net with 2 hidden layers
weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer_size])
outputs_3 = tf.nn.relu(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("weights_4", [hidden_layer_size, hidden_layer_size])
biases_4 = tf.get_variable("biases_4", [hidden_layer_size])
outputs_4 = tf.nn.relu(tf.matmul(outputs_3, weights_4) + biases_4)

weights_5 = tf.get_variable("weights_5", [hidden_layer_size, hidden_layer_size])
biases_5 = tf.get_variable("biases_5", [hidden_layer_size])
outputs_5 = tf.nn.relu(tf.matmul(outputs_4, weights_5) + biases_5)

weights_6 = tf.get_variable("weights_6", [hidden_layer_size, hidden_layer_size])
biases_6 = tf.get_variable("biases_6", [hidden_layer_size])
outputs_6 = tf.nn.relu(tf.matmul(outputs_5, weights_6) + biases_6)

weights_7 = tf.get_variable("weights_7", [hidden_layer_size, hidden_layer_size])
biases_7 = tf.get_variable("biases_7", [hidden_layer_size])
outputs_7 = tf.nn.relu(tf.matmul(outputs_6, weights_7) + biases_7)



weights_8 = tf.get_variable("weights_8", [hidden_layer_size, output_size])
biases_8 = tf.get_variable("biases_8", [output_size])
# We will incorporate the softmax activation into the loss as in the previous example
outputs = tf.matmul(outputs_7, weights_8) + biases_8

# Use the softmax cross entropy loss with logits
loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)

# Get a 0 or 1 for every input indicating whether it output the correct answer
out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

# Optimize with Adam
optimize = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mean_loss)

# Create a session
sess = tf.InteractiveSession()


# Initialize the variables
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Choose the batch size
batch_size = 150

# Set early stopping mechanisms
max_epochs = 25
prev_validation_loss = 9999999.

# Load the first batch of training and validation using the class we created. 
# Arguments are ending of 'Audiobooks_Data_<...>' where for <...> we input 'train' 'validation' or 'test'
# depending on what we want to load
train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')


# Create the loop for epochs 
for epoch_counter in range(max_epochs):
 
 # Set the epoch loss to 0, and make it a float
 curr_epoch_loss = 0.
 
 # Iterate over the training data 
 # Since train_data is an instance of the Audiobooks_Data_Reader class,
 # we can iterate through it by implicitly using the __next__ method we defined above.
 # As a reminder, it batches samples together, one-hot encodes the targets, and returns
 # inputs and targets batch by batch
 for input_batch, target_batch in train_data:
	 _, batch_loss = sess.run([optimize, mean_loss], 
		 feed_dict={inputs: input_batch, targets: target_batch})
	 
	 #Record the batch loss into the current epoch loss
	 curr_epoch_loss += batch_loss
 
 # Find the mean curr_epoch_loss
 # batch_count is a variable, defined in the Audiobooks_Data_Reader class
 curr_epoch_loss /= train_data.batch_count
 
 # Set validation loss and accuracy for the epoch to zero
 validation_loss = 0.
 validation_accuracy = 0.
 
 # Use the same logic of the code to forward propagate the validation set
 # There will be a single batch, as the class was created in this way
 for input_batch, target_batch in validation_data:
	 validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
		 feed_dict={inputs: input_batch, targets: target_batch})
 
 # Print statistics for the current epoch
 print('Epoch '+str(epoch_counter+1)+
	   '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
	   '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
	   '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
 
 # Trigger early stopping if validation loss begins increasing.,
 if validation_loss > prev_validation_loss:
	 break
	 
 # Store this epoch's validation loss to be used as previous in the next iteration.,
 prev_validation_loss = validation_loss
 
print('End of training.')


test_data = Audiobooks_Data_Reader('test')

for input_batch, target_batch in test_data:
    test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})
    
test_accuracy_percent = test_accuracy[0] * 100.
# Print the test accuracy formatted in percentages
print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')

