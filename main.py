import tensorflow as tf
import src.began

with tf.Session() as sess:
    model = src.began.Began(sess)
    model.train()