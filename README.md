# train_tf
regressionex1: 
  - basic regression example using tensor flow low level api
  - save/load a model

cnn
 cifar
     The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
     There are 50000 training images and 10000 test images.     
     The dataset is divided into five training batches and one test batch, each with 10000 images. 
     The test batch contains exactly 1000 randomly-selected images from each class. 
     The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. 
     Between them, the training batches contain exactly 5000 images from each class.
         un/pickle - object from/to stream(binary files)
		 
	The final model is saved in path: /tmp
	
	the code is split between - distinct class for data manipulation and distinct class for model/estimator buil, train, evaluate and predict
	
estimatorAPI
    regression
     tensExercise
         pandas df manipulation
         DNNRegressor
         sklearn.model_selection  train_test_split,  MinMaxScaler
     
	 
	 
	 https://karpathy.github.io/2015/05/21/rnn-effectiveness/