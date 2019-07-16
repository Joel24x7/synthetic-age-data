import tensorflow as tf
import numpy as np
import PIL.Image as PILImage
import scipy
import matplotlib.pyplot as plt
from hyperparameters import *
from models import *
import mnist_preprocess
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
sess=tf.Session()

x_placeholder = tf.placeholder('float', shape=[batch_size, image_size, image_size, 3])
z_placeholder = tf.placeholder(tf.float32, [None, noise_dimension])

print('Checkpoint 1: setup')

Dx = forward_pass_discriminator(x_placeholder)
Gz = forward_pass_generator(z_placeholder)
Dg = forward_pass_discriminator(Gz, reuse='False')

print('Checkpoint 2: Models built')

real_image_loss = l1_loss(x_placeholder, Dx)
fake_image_loss = l1_loss(Gz, Dg)
discriminator_loss = discriminator_loss(real_image_loss, fake_image_loss, kt_equilbrium_term)
generator_loss = fake_image_loss
convergence = convergence(real_image_loss, fake_image_loss, gamma_diversity_ratio)

print('Checkpoint 3: Loss defined')

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dec' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

adam = tf.train.AdamOptimizer(learning_rate)
dis_opt = adam.minimize(discriminator_loss, var_list=d_vars)
gen_opt = adam.minimize(generator_loss, var_list=g_vars)

print('Checkpoint 4: Optimizer defined')

sess.run(tf.global_variables_initializer())
iterations=2
for i in range(iterations):
    z_batch=np.random.uniform(-1,1,size=[batch_size, noise_dimension])
    real_image_batch=load()
    _, dloss=sess.run([dis_opt, discriminator_loss], feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch})
    _, gloss=sess.run([gen_opt, generator_loss], feed_dict={z_placeholder: z_batch})
    print('Step {} - Losses: {} {}'.format(i, dloss,  gloss))

sample_image=forward_pass_generator(z_placeholder, reuse=True)
z_batch = np.random.uniform(-1,1,size=[1, noise_dimension])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i)
plt.show()
