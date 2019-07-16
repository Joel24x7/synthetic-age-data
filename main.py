import tensorflow as tf
from models import *
from wrapped_utils import *
from hyperparameters import *

def batch_step():
    #Define what happens for each batch of data
    pass

def train():
    loss_tracker={'generator':[], 'discriminator':[], 'convergence':[]}

    graph=tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initailzer=tf.constant_intializer(0), trainable=False)

        with tf.device('/gpu:0'):
            adam = tf.train.AdamOptimizer(learning_rate=learning_rate)





