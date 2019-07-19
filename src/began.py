import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from datasets.mnist_preprocess import load_data
from src.config import *
from src.models import *
from src.utils import *


class Began():
    def __init__(self, sess):

        self.sess = sess

        if not os.path.exists('assets'):
            os.makedirs('assets')
        make_file_structure(project_dir)

        self.compile_model()

    def load(self, sess, saver):
        print("\nLoading checkpoints...\n")
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_name))

    def compile_model(self):
        #Inputs
        self.x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='x')
        self.z = tf.placeholder(tf.float32, [batch_size, noise_dimension], name='z')
        self.kt = tf.placeholder(tf.float32, name='kt')
        self.lr = tf.placeholder(tf.float32, name='lr')

        '''
        Forward Pass

        G(z) - Generator Output of Random Noise
        D(x) - Discriminator Output of Training Data
        D(z) - Discriminator Output of Generator Output
        '''
        self.g_z = forward_pass_generator(self.z)
        self.d_x = forward_pass_discriminator(self.x)
        self.d_z = forward_pass_discriminator(self.g_z, reuse=True)

        '''
        Loss - BEGAN Objective

        Ld = L(x) - kt * L(G(z))
        Lg = L(G(z))
        *kt+1 = kt + lamda_kt * (gamma * L(x) - L(G(z)))*
        m_global = L(x) + |gamma * L(x) - L(G(z))|
        '''
        self.d_x_loss = l1_loss(self.x, self.d_x)
        self.d_z_loss = l1_loss(self.g_z, self.d_z)
        self.dis_loss = self.d_x_loss - self.kt * self.d_z_loss
        self.gen_loss = self.d_z_loss
        self.convergence = self.d_x_loss + tf.abs(diversity_ratio * self.d_x_loss - self.d_z_loss)

        #Variables
        gen_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "gen")
        dis_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "dis")

        '''
        Backward Pass and Parameter Updates
        
        Using Adam Optimizer with learning rate = 1e-4
        '''
        self.gen_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.gen_loss, var_list=gen_variables)
        self.dis_optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.dis_loss, var_list=dis_variables)

        #Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        #Save weights
        self.saver = tf.train.Saver()
        

        try:
            self.load(self.sess, self.saver)
        except:
            self.saver.save(self.sess, model_name, write_meta_graph=True)

        '''
        Summary

        Monitor training progress
        '''
        tf.summary.scalar('loss/gen_loss', self.gen_loss)
        tf.summary.scalar('loss/dis_loss', self.dis_loss)
        tf.summary.scalar('loss/kt', self.kt)
        tf.summary.scalar('loss/convergence', self.convergence)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(project_dir, self.sess.graph)

    def train(self):

        print("\nBeginning training...\n")
        data = load_data()
        np.random.shuffle(data)
        start_time = time.time()
        kt = kt_config
        lr = learning_rate


        num_batches_per_epoch = len(data) // batch_size
        self.count = 0

        for epoch in range(epochs):

            for batch_step in range(num_batches_per_epoch):
                self.count += 1
                
                #Prep training (x) and noise (z) batches
                start_data_batch = batch_step * batch_size
                end_data_batch = start_data_batch + batch_size
                batch_data = data[start_data_batch:end_data_batch, :, :, :]
                z_batch = np.random.uniform(-1,1,size=[batch_size, noise_dimension])

                #Prep tf fetches and feed dictionary
                gen_fetches = [self.gen_optimizer, self.gen_loss, self.d_x_loss, self.d_z_loss]
                dis_fetches = [self.dis_optimizer, self.dis_loss, self.merged]
                feed_dict = {self.x: batch_data, self.z: z_batch, self.kt: kt, self.lr: learning_rate}

                #Run tf session
                _, gen_loss_output, d_x_loss_output, d_z_loss_output = self.sess.run(gen_fetches, feed_dict=feed_dict)
                _, dis_loss_output, summary = self.sess.run(dis_fetches, feed_dict=feed_dict)

                #Update dynamic variables (kt and convergence)
                kt = np.maximum(np.minimum(1., kt + lambda_kt * (diversity_ratio * d_x_loss_output - d_z_loss_output)), 0.)
                convergence = d_x_loss_output + np.abs(diversity_ratio * d_x_loss_output - d_z_loss_output)
                loss = gen_loss_output + dis_loss_output
               
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, "
                      "loss: %.4f, loss_g: %.4f, loss_d: %.4f, d_real: %.4f, d_fake: %.4f, kt: %.8f, M: %.8f"
                      % (epoch, batch_step, num_batches_per_epoch, time.time() - start_time,
                         loss, gen_loss_output, dis_loss_output, d_x_loss_output, d_z_loss_output, kt, convergence))

                self.writer.add_summary(summary, self.count)

                # Test during Training
                if self.count % snapshot == (snapshot - 1):
                    # update learning rate
                    lr *= 0.95
                    # save & test
                    self.saver.save(self.sess, model_name, global_step=self.count, write_meta_graph=False)
                    self.test('train')

    
    def test(self, key='test'):

        #Generate output
        num_images = batch_size
        
        z_test = np.random.uniform(-1,1,size=[batch_size, noise_dimension])
        output_gen = (self.sess.run(self.g_z, feed_dict={self.z : z_test}))

        for i in range(num_images):
            tmpName = 'assets/mnist_model/results/{}_image{}.png'.format(key, i)
            img = output_gen[i]
            plt.imshow(img)
            plt.savefig(tmpName)