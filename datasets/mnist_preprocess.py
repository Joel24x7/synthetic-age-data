import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as PILImage
import scipy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

image_size=64

def load_data():
    with h5py.File('datasets/mnist_data.h5') as file:
        data = file['mnist_data']
        data = np.array(data, dtype=np.float16)
        return data

def prep_mnist_color(change_colors=True):

    print("\nPreparing Mnist digits...\n")

    scale_factor=image_size/28.0
    x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    
    path = os.path.abspath('color_img.png') 
    color_img = PILImage.open(path)

    # Resize (this is optional but results in a training set of larger images)
    resized = np.asarray([scipy.ndimage.zoom(image, (scale_factor, scale_factor, 1), order=1) for image in x_train])
    
    # Extend to RGB
    color = np.concatenate([resized, resized, resized], axis=3)
    
    # Convert the MNIST images to binary
    binary = (color > 0.5)
    
    # Create a new placeholder variable for our data
    dataset = np.zeros((x_train.shape[0], image_size, image_size, 3))
    for i in range(x_train.shape[0]):
        # Take a random crop of the color_img image (background)
        x_c = np.random.randint(0, color_img.size[0] - image_size)
        y_c = np.random.randint(0, color_img.size[1] - image_size)
        image = color_img.crop((x_c, y_c, x_c + image_size, y_c + image_size))
        # Conver the image to float between -1 and 1
        image = np.asarray(image) / (255*0.5)
        image -= 1.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[binary[i]] = 1 - image[binary[i]]
        
        dataset[i] = image
    print("\nPreprocessing successful\n")
    return dataset

if __name__ == "__main__":
    data = prep_mnist_color()
    with h5py.File('mnist_data.h5', 'w') as file:
        file.create_dataset('mnist_data', data=data)