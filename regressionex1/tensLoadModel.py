# -*- coding: utf-8 -*-
import tensorflow as tf

tf.reset_default_graph()

w = tf.Variable(0.5)
b = tf.Variable(0.5)

saver = tf.train.Saver()

with tf.Session() as sess:
    
    # Restore the model
    saver.restore(sess,'models/6_model.ckpt')   

    # Fetch Back Results
    restored_w , restored_b = sess.run([w,b])
    print("slope: ", restored_w, "intercept", restored_b)