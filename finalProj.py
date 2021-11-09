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
poses = []

for pose in os.listdir(train_dir):
    print(pose)
    currdir = train_dir + '/' + pose

    for image in os.listdir(currdir):
        poses.append(pose) #Appends the pose to keep the same order as the images
        im = io.imread(train_dir + '/' + pose + '/' + image)
        grayIm = skimage.color.rgb2gray(im)
        trainim.append(im)
        print(im)
        yenned = filters.threshold_yen(grayIm)
        binary = grayIm > yenned
        print(binary)
        newimage = np.ndarray(shape = (binary.shape[0], binary.shape[1], 3), dtype = np.uint8)
        for row in range(len(binary)):
          for pixel in range(len(binary[row])):
            if binary[row][pixel] == True:
              newimage[row][pixel] = [255, 255, 255]
            else:
              newimage[row][pixel] = [0, 0, 0]
        plt.figure()
        plt.imshow(newimage)
        break

for ax, im, p in zip(grid, trainim, poses):
  ax.imshow(im)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.grid(False)
  ax.set_title(p)

plt.show()

