import tensorflow as tf
import numpy as np
import PIL.Image as PILImage
import scipy
import matplotlib.pyplot as plt
from hyperparameters import *
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist_color(change_colors=True):

    scale_factor=image_size/28.0
    x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    lena = PILImage.open('lena.png')

    # Resize (this is optional but results in a training set of larger images)
    resized = np.asarray([scipy.ndimage.zoom(image, (scale_factor, scale_factor, 1), order=1) for image in x_train])
    
    # Extend to RGB
    color = np.concatenate([resized, resized, resized], axis=3)
    
    # Convert the MNIST images to binary
    binary = (color > 0.5)
    
    # Create a new placeholder variable for our data
    dataset = np.zeros((x_train.shape[0], image_size, image_size, 3))
    
    for i in range(x_train.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - image_size)
        y_c = np.random.randint(0, lena.size[1] - image_size)
        image = lena.crop((x_c, y_c, x_c + image_size, y_c + image_size))
        # Conver the image to float between 0 and 1
        image = np.asarray(image) / 255.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[binary[i]] = 1 - image[binary[i]]
        
        dataset[i] = image
    return dataset