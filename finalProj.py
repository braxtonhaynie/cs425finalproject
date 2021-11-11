from itertools import count
from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io
from skimage import filters
from skimage import restoration
import pandas as pd
import skimage
import os



train_dir = 'DATASET/TRAIN'
test_dir = 'DATASET/TEST'

fig = plt.figure(figsize = (5, 10))
grid = ImageGrid(fig, 111, nrows_ncols =(5, 1), axes_pad = 0.4)
trainim = []
trainposes = []

for pose in os.listdir(train_dir):
  print(pose)
  currdir = train_dir + '/' + pose
  i = 0
  for image in os.listdir(currdir):
    trainposes.append(pose) #Appends the pose to keep the same order as the images
    im = io.imread(train_dir + '/' + pose + '/' + image)
    grayIm = skimage.color.rgb2gray(im)
    # trainim.append(im)
    yenned = filters.threshold_yen(grayIm)
    binary = grayIm > yenned
    newimage = np.ndarray(shape = (binary.shape[0], binary.shape[1], 3), dtype = np.uint8)
    nb = []
    for row in range(len(binary)):
      # nb = nb.join(binary[row])
      for pixel in range(len(binary[row])):
        nb.append(1 if binary[row][pixel] is True else 0)
      #   if binary[row][pixel] == True:
      #     newimage[row][pixel] = [255, 255, 255]
      #   else:
      #     newimage[row][pixel] = [0, 0, 0]
    # plt.figure()
    # plt.imshow(newimage)

    trainim.append(nb)
    if i == 2:
      break
    else:
      i += 1


testim = []
testposes = []
for pose in os.listdir(test_dir):
  print(pose)
  currdir = train_dir + '/' + pose
  i = 0
  for image in os.listdir(currdir):
    testposes.append(pose)
    im = io.imread(train_dir + '/' + pose + '/' + image)
    grayIm = skimage.color.rgb2gray(im)
    yenned = filters.threshold_yen(grayIm)
    binary = grayIm > yenned
    newimage = np.ndarray(shape=(binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    nb = []
    for row in range(len(binary)):
      # nb = nb.join(binary[row])
      for pixel in range(len(binary[row])):
        nb.append(1 if binary[row][pixel] is True else 0)

      #   if binary[row][pixel] == True:
      #     newimage[row][pixel] = [255, 255, 255]
      #   else:
      #     newimage[row][pixel] = [0, 0, 0]
    # plt.figure()
    # plt.imshow(newimage)

    testim.append(nb)
    if i == 2:
      break
    else:
      i += 1


# for ax, im, p in zip(grid, trainim, trainposes):
#   ax.imshow(im)
#   ax.set_xticks([])
#   ax.set_yticks([])
#   ax.grid(False)
#   ax.set_title(p)

# plt.show()

from sklearn.neural_network import MLPClassifier

net = MLPClassifier()
print(trainim[1])
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