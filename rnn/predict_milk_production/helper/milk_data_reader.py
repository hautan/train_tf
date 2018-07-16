import numpy as np

class Milk_Data_Reader():
    
    def next_batch(training_data,batch_size,steps):
    	"""
    	INPUT: Data, Batch Size, Time Steps per batch
    	OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    	"""
    	
    	# STEP 1: Use np.random.randint to set a random starting point index for the batch.
    	# Remember that each batch needs have the same number of steps in it.
    	# This means you should limit the starting point to len(data)-steps
       # Grab a random starting point for each batch
    	rand_start = np.random.randint(0,len(training_data)-steps) 
    	
    	# STEP 2: Now that you have a starting index you'll need to index the data from
    	# the random start to random start + steps. Then reshape this data to be (1,steps)
        
       # Create Y data for time series in the batches
    	y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    	
    	# STEP 3: Return the batches. You'll have two batches to return y[:,:-1] and y[:,1:]
    	# You'll need to reshape these into tensors for the RNN. Depending on your indexing it
    	# will be either .reshape(-1,steps-1,1) or .reshape(-1,steps,1)              	
    
    	return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)
    
    def __iter__(self):
         return self

    