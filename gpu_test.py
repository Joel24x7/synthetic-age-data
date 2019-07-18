import tensorflow as tf
import src.began

with tf.device('/gpu:0'):
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        model = src.began.Began(sess)
        model.train()