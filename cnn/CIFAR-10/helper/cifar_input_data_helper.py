import numpy as np
from helper.utils import one_hot_encode 
import tensorflow as tf
import constants as ct

class CifarInputDataHelper():
    
#    def __init__(self, data_batch1, data_batch2, data_batch3, data_batch4, data_batch5, test_batch):
    def __init__(self):
        self.i = 0
        
        # Grabs a list of all the data batches for training
#        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
#        self.all_train_batches = data_batch
        # Grabs a list of all the test batches (really just one batch)
#        self.test_batch = [test_batch]
#        self.test_batch = test_batch
        self.import_data()
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
#        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)        
#        self.training_labels = tf.one_hot(indices=tf.cast(np.hstack([d[b"labels"] for d in self.all_train_batches]), tf.int32), depth=10)
#        self.training_labels = tf.cast(np.hstack([d[b"labels"] for d in self.all_train_batches]), tf.int32)
        self.training_labels = np.hstack([d[b"labels"] for d in self.all_train_batches])
        
        print("Setting Up Test Images and Labels")
        
        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
#        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)
#        self.test_labels = tf.one_hot(indices=tf.cast(np.hstack([d[b"labels"] for d in self.test_batch]), tf.int32), depth=10)
#        self.test_labels = tf.cast(np.hstack([d[b"labels"] for d in self.test_batch]), tf.int32)
        self.test_labels = np.hstack([d[b"labels"] for d in self.test_batch])

        
    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y    
    
    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict

    def import_data(self):
        dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
        all_data = [0,1,2,3,4,5,6]
        for i,direc in zip(all_data,dirs):
            all_data[i] = self.unpickle(ct.CIFAR_DIR+direc)
            
        batch_meta = all_data[0]
        data_batch1 = all_data[1]
        data_batch2 = all_data[2]
        data_batch3 = all_data[3]
        data_batch4 = all_data[4]
        data_batch5 = all_data[5]
        test_batch = all_data[6]
        
        data_batch = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        test_batch = [test_batch]
        
        self.all_train_batches = data_batch
        self.test_batch = test_batch
        self.batch_meta = batch_meta