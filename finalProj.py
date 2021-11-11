from itertools import count
from math import inf
from ssl import PEM_cert_to_DER_cert
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.core.fromnumeric import resize, shape, std
import pandas as pd
import skimage
import os
from skimage import io
from skimage import filters
from skimage import restoration
from skimage import color
from skimage.transform import resize
from PIL import ImageFile

# from sklearn.

ImageFile.LOAD_TRUNCATED_IMAGES = True

maxImages = 5

imsize = (500, 500)
train_dir = 'DATASET/TRAIN'
test_dir = 'DATASET/TEST'


def read_images(dir):
  """
  Reads all images in a directory and returns them as a list of numpy arrays.
  """
  y = []
  X = []
  for pose in os.listdir(dir):
    print(pose)
    currdir = train_dir + '/' + pose
    i = 0
    for image in os.listdir(currdir):
      print(" ({})".format(image))
      y.append(pose) #Appends the pose to keep the same order as the images    
      im = io.imread(train_dir + '/' + pose + '/' + image)    

      print(shape(im), file=os.sys.stderr)
      # Converts the image to RGB (From grayscale or RGBA)
      if len(shape(im)) == 3:
        if shape(im)[2] == 4:
          im = color.rgba2rgb(im)
      elif len(shape(im)) == 2:
        im = color.gray2rgb(im)
      #Resizes the image to 500x500
      im = resize(im, imsize)
      # grayIm = skimage.color.rgb2gray(im)
      yenned = filters.threshold_yen(im)
      binary = im > yenned
      nb = []
      for row in range(len(binary)):
        for pixel in range(len(binary[row])):
          nb.append(1 if binary[row][pixel] is True else 0)
      X.append(nb)
      i += 1
      if i == maxImages:
        break
  return X, y
    

fig = plt.figure(figsize = (5, 10))
grid = ImageGrid(fig, 111, nrows_ncols =(5, 1), axes_pad = 0.4)
trainim = []
trainposes = []
trainim, trainposes = read_images(train_dir)


testim = []
testposes = []
testim, testposes = read_images(test_dir)

# for ax, im, p in zip(grid, trainim, trainposes):
#   ax.imshow(im)
#   ax.set_xticks([])
#   ax.set_yticks([])
#   ax.grid(False)
#   ax.set_title(p)

# plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

parameters = [
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'momentum': [0.1, 0.9],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [524],
    'early_stopping': [False],
    'max_iter': [10000],
  }, 
  { 'hidden_layer_sizes': [8, (8,4), (8,8), 16, (16,8)],
    'activation': ['tanh'], 
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'momentum': [0.1, 0.9],
    'nesterovs_momentum': [True],
    'shuffle': [True],
    'random_state': [524],
    'early_stopping': [False],
    'max_iter': [10000],
  }
]

net = GridSearchCV(MLPClassifier(),param_grid=parameters, n_jobs=4)

print(np.shape(trainim))
print(np.shape(trainim[0]))
trainim = np.matrix(trainim, dtype=np.int8)
trainposes = np.array(trainposes)
testim = np.matrix(testim, dtype=np.int8)
print(trainim)
print(trainposes)
net.fit(trainim, trainposes)
ypred = net.predict(testim)

count = 0
correct = 0
for i in range(len(ypred)):
  if ypred[i] == testposes[i]:
    print('Correct: {}'.format(testposes[i]))
    correct += 1
  else:
    print('Incorrect: Guessed {}, Correct: {}'.format(ypred[i], testposes[i]))
  count += 1

print()
print('Accuracy: {}'.format(correct/count))
print("{} / {}".format(correct, count))

