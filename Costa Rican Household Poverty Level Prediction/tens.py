# -*- coding: utf-8 -*-
 # We must always import the relevant libraries for our problem at hand. NumPy and TensorFlow are required for this example.
# https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data#_=_
import numpy as np
np.set_printoptions(threshold='nan')
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

def toInt(x):
    if x == 'yes':
        return 1
    else:
        if x == 'no':
            return 0
        else:
            return x

costa_rica_household = pd.read_csv('data/train.csv')

#x1 = 
costa_rica_household.describe()
#x1["v2a1"]
costa_rica_household.head()
list(costa_rica_household.dtypes)

#costa_rica_household = costa_rica_household.fillna(0)
costa_rica_household = costa_rica_household.fillna(costa_rica_household.mean())
#costa_rica_household["idhogar"] = costa_rica_household["idhogar"].apply(lambda x: int(x, 16))
#costa_rica_household["dependency"] = costa_rica_household["dependency"].apply(lambda x: toInt(x))
#costa_rica_household["edjefe"] = costa_rica_household["edjefe"].apply(lambda x: toInt(x))//edjefa

#costa_rica_household.loc[costa_rica_household['dependency'] == "'<='"]
#v1 = costa_rica_household[costa_rica_household['dependency'].apply(lambda x: type(x) == str)]['dependency']



#col_name = costa_rica_household.columns
#print(list(col_name))
#costa_rica_household[["age", "SQBage", "agesq", "r4h1", "r4h2"]]

cols_to_norm = ['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 
                'tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 
                'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 
                'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3',
                'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
                'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis', 'male', 'female', 
                'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 
                'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 
                'parentesco12', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'meaneduc', 'instlevel1', 
                'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 
                'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4',
                'lugar5', 'lugar6', 'area1', 'area2', 'SQBescolari', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency',
                'SQBmeaned', 'agesq']
cat_cols_to_norm = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']
cols_of_interest = ['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3',
                'tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 
                'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 
                'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3',
                'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4',
                'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis', 'male', 'female', 
                'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 
                'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 
                'parentesco12', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'meaneduc', 'instlevel1', 
                'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 
                'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4',
                'lugar5', 'lugar6', 'area1', 'area2', 'SQBescolari', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency',
                'SQBmeaned', 'agesq']



#costa_rica_household[cols_to_norm] = costa_rica_household[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
#costa_rica_household[cat_cols_to_norm] = costa_rica_household[cat_cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
costa_rica_household[cols_of_interest] = costa_rica_household[cols_of_interest].apply(lambda x: (x - x.min())/(x.max() - x.min()))

feat_cols = []
for col_name in cols_to_norm:
    col_name = tf.feature_column.numeric_column(col_name)
    feat_cols.append(col_name)

age_range_count = [1,2,3,4,5,7]

r4h1_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4h1'), boundaries=age_range_count)
r4h2_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4h2'), boundaries=age_range_count)
r4h3_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4h3'), boundaries=age_range_count)
crossed_r4h = tf.feature_column.crossed_column([r4h1_bucket, r4h2_bucket, r4h3_bucket], 100)
#fc = [r4h1_bucket, r4h2_bucket, r4h3_bucket, crossed_r4h]

r4m1_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4m1'), boundaries=age_range_count)
r4m2_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4m2'), boundaries=age_range_count)
r4m3_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4m3'), boundaries=age_range_count)
crossed_r4m = tf.feature_column.crossed_column([r4m1_bucket, r4m2_bucket, r4m3_bucket], 100)

r4t1_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4t1'), boundaries=age_range_count)
r4t2_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4t2'), boundaries=age_range_count)
r4t3_bucket = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('r4t3'), boundaries=age_range_count)
crossed_r4t = tf.feature_column.crossed_column([r4t1_bucket, r4t2_bucket, r4t3_bucket], 100)

feat_cols.extend([r4h1_bucket, r4h2_bucket, r4h3_bucket, crossed_r4h, r4m1_bucket, r4m2_bucket, r4m3_bucket, crossed_r4m, r4t1_bucket, r4t2_bucket, r4t3_bucket, crossed_r4t])

len(feat_cols)
feat_cols[138]


estimator = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=4)
#costa_rica_household[(costa_rica_household.Target == 4)]

x_data = costa_rica_household.drop('Id', axis=1).drop('edjefa', axis=1).drop('idhogar', axis=1).drop('dependency', axis=1).drop('Target', axis=1)
#x_data['idhogar']
#x_data.describe()
#x_data.head()
labels = costa_rica_household['Target']
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
