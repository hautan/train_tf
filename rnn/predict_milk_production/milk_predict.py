#/notebooks/03-Time-Series-Exercise-Solutions-Final.ipynb

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper.milk_data_reader import Milk_Data_Reader
import helper.constants as ct

tf.logging.set_verbosity(tf.logging.INFO)
milk = pd.read_csv('data/monthly-milk-production.csv', index_col='Month')
milk.head()
milk.describe()

milk.index = pd.to_datetime(milk.index)

milk.describe().transpose()
milk.index
milk.plot()
milk.info()
#Create a test train split using indexing (hint: use .head() or tail() or .iloc[]). We don't want a random train test split, we want to specify that the test set is the last 3 months of data 
#is the test set, with everything before it is the training.
train_set = milk.head(156)
test_set = milk.tail(12)

#scala the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_set)

train_scaled = scaler.transform(train_set)
test_scaled = scaler.transform(test_set)

X = tf.placeholder(tf.float32, [None, ct.num_time_steps, ct.num_inputs])
y = tf.placeholder(tf.float32, [None, ct.num_time_steps, ct.num_outputs])

# Also play around with GRUCell
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=ct.num_neurons, activation=tf.nn.relu), output_size=ct.num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#Loss Function and Optimizer
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=ct.learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#with tf.Session(config=tf.ConfigProto(gpu_options=ct.gpu_options)) as sess:
with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(ct.num_train_iterations):        
        X_batch, y_batch = Milk_Data_Reader.next_batch(train_scaled,ct.batch_size,ct.num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./ex_time_series_model")
    
test_set

with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # Create a numpy array for your genreative seed from the last 12 months of the 
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-12:])
    
    ## Now create a for loop that 
    for iteration in range(12):
        X_batch = np.array(train_seed[-ct.num_time_steps:]).reshape(1, ct.num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])

train_seed
#Grab the portion of the results that are the generated values and apply inverse_transform on them to turn them back into milk production value units (lbs per cow). 
#Also reshape the results to be (12,1) so we can easily add them to the test_set dataframe.
results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

#Create a new column on the test_set called "Generated" and set it equal to the generated results.
test_set['Generated'] = results
test_set
test_set.plot()