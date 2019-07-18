from datasets.mnist_preprocess import load_data
import matplotlib.pyplot as plt
import numpy as np

test = np.random.random((64,64,3))
plt.imshow(test)
plt.savefig('test')
