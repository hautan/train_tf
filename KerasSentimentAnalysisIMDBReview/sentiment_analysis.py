import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb 
import matplotlib.pyplot as plt

#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

top_words = 5000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

len(X_train[0])
len(X_train[1])
len(X_train[2])


#NUM_WORDS=1000 # only use top 1000 words
#INDEX_FROM=3   # word index offset
#
##reverse lookup
#word_to_id = keras.datasets.imdb.get_word_index()    
#word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
#    
#word_to_id["<PAD>"] = 0
#word_to_id["<START>"] = 1
#word_to_id["<UNK>"] = 2
#id_to_word = {value:key for key,value in word_to_id.items()}
#print(' '.join(id_to_word[id] for id in X_train[0] ))

# Truncate and pad the review sequences 
from keras.preprocessing import sequence 
max_review_length = 500 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Build the model 
embedding_vector_length = 32 
model = Sequential() 
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length)) 
model.add(LSTM(100)) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 

#Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64) 

#sudo tensorboard --logdir=/tmp

#Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))

##predict sentiment from reviews
#bad = "this movie was terrible and bad"
#good = "i really liked the movie and had fun"
#for review in [good,bad]:
#    tmp = []
#    for word in review.split(" "):
#        tmp.append(word_to_id[word])
#    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
#    print("%s. Sentiment: %s" % (review,model.predict(array([tmp_padded][0]))[0][0]))

