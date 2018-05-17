# -*- coding: utf-8 -*-
 # We must always import the relevant libraries for our problem at hand. NumPy and TensorFlow are required for this example.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

x_data = np.linspace(0.0, 10.0, 1000000)
y_data = (0.5*x_data) + 5
x_df = pd.DataFrame(data=x_data, columns=['X data'])
y_df = pd.DataFrame(data=x_data, columns=['Y'])
my_data = pd.concat([x_df, y_df], axis=1)

feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
print(x_train.shape, y_eval.shape)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func,steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

print('Train metrics')
print(train_metrics)
print('Eval metrics')
print(eval_metrics)

brand_new_data = np.linspace(0.0, 10.0, 100)
pred_input_func = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle=True)
predictions = []
for predict in estimator.predict(input_fn=pred_input_func):
    predictions.append(predict)
    

sample_data = my_data.sample(n=100)
sample_data.plot(kind='scatter', x='X data', y='Y')
plt.plot(brand_new_data, predictions, 'r*')
