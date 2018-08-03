#one hot encode your documents
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.preprocessing.text import one_hot
docs = ['Gut gemacht',
        'Gute arbeit',
        'Super idee',
        'Perfekt erledigt',
        'exzellent',
        'naja',
        'Schwache arbeit.',
        'Nicht gut',
        'Miese arbeit.',
        'Hätte es besser machen können.']
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)