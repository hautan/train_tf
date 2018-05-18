# -*- coding: utf-8 -*-
 # We must always import the relevant libraries for our problem at hand. NumPy and TensorFlow are required for this example.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

diabetes = pd.read_csv('diabetes.csv')

diabetes.describe()
diabetes.head()

print(diabetes.columns)
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))

pregnancies = tf.feature_column.numeric_column('Pregnancies')
glucose = tf.feature_column.numeric_column('Glucose')
blood_pressure = tf.feature_column.numeric_column('BloodPressure')
skin_thickness = tf.feature_column.numeric_column('SkinThickness')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree_function = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')

#categorical_columun_voc = tf.feature_column.categorical_column_with_vocabulary_list('GroupColName',['a','b','c'])
#categorical_columun = tf.feature_column.categorical_column_with_hash_bucket('GroupColName', hashbucketsize=10)

#diabetes['Age'].hist(bins=10)
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [pregnancies, glucose, blood_pressure, skin_thickness, insulin,bmi, diabetes_pedigree_function, age_bucket]
estimator = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

x_data = diabetes.drop('Outcome', axis=1)
x_data.head()
labels = diabetes['Outcome']
labels.head()

from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(x_data, labels, test_size=0.3, random_state=101)
print(X_train.shape, y_eval.shape)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=100, shuffle=True)
estimator.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_eval, y=y_eval, batch_size=10, num_epochs=1, shuffle=False)
eval_metrics = estimator.evaluate(input_fn=eval_input_func)


print('Eval metrics')
print(eval_metrics)


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_eval, shuffle=False)
predictions = []
for predict in estimator.predict(input_fn=pred_input_func):
    predictions.append(predict)

predictions

#categorical_columun_voc = tf.feature_column.embedding_column(categorical_columun_voc, 4)
dnn_classifier = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_classifier.train(input_fn=input_func,steps=1000)
dnn_eval_metrics = dnn_classifier.evaluate(input_fn=eval_input_func)

dnn_eval_metrics
