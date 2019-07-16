import tensorflow as tf
import numpy as np
import PIL.Image as PILImage
import scipy
import matplotlib.pyplot as plt
from hyperparameters import *
from models import *
from tensorflow.examples.tutorials.mnist import input_data


x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
lena = PILImage.open('lena.png')

def get_mnist_batch(change_colors=True):
    
    # Select random batch (WxHxC)
    idx = np.random.choice(x_train.shape[0], batch_size)
    batch_raw = x_train[idx, :, :, 0].reshape((batch_size, 28, 28, 1))
    
    # Resize (this is optional but results in a training set of larger images)
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (4.57, 4.57, 1), order=1) for image in batch_raw])
    
    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)
    
    # Convert the MNIST images to binary
    batch_binary = (batch_rgb > 0.5)
    
    # Create a new placeholder variable for our batch
    batch = np.zeros((batch_size, 128, 128, 3))
    
    for i in range(batch_size):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 128)
        y_c = np.random.randint(0, lena.size[1] - 128)
        image = lena.crop((x_c, y_c, x_c + 128, y_c + 128))
        # Conver the image to float between 0 and 1
        image = np.asarray(image) / 255.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]
        
        batch[i] = image

    return batch


# examples=get_mnist_batch()
# count = 16
# plt.figure(figsize=(15,3))
# for i in range(count):
#     plt.subplot(2, count // 2, i+1)
#     plt.imshow(examples[i])
#     plt.axis('off')
    
# plt.tight_layout()
# plt.show()

tf.reset_default_graph()
sess=tf.Session()

x_placeholder = tf.placeholder('float', shape=[batch_size, image_size, image_size, 3])
z_placeholder = tf.placeholder(tf.float32, [None, noise_dimension])

print('Checkpoint 1: setup')

Dx = build_discriminator(x_placeholder)
Gz = build_generator(z_placeholder)
Dg = build_discriminator(Gz, reuse='False')

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
    real_image_batch=get_mnist_batch()
    _, dloss=sess.run([dis_opt, discriminator_loss], feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch})
    _, gloss=sess.run([gen_opt, generator_loss], feed_dict={z_placeholder: z_batch})
    print('Step {} - Losses: {} {}'.format(i, dloss,  gloss))

sample_image=build_generator(z_placeholder, reuse=True)
z_batch = np.random.uniform(-1,1,size=[1, noise_dimension])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i)
plt.show()
