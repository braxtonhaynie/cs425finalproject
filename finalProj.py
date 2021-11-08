from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io
import pandas as pd

import os

train_dir = 'DATASET/TRAIN'
test_dir = 'DATASET/TEST'

fig = plt.figure(figsize = (5, 10))
grid = ImageGrid(fig, 111, nrows_ncols =(5, 1), axes_pad = 0.4)
trainim = []
poses = []

for pose in os.listdir(train_dir):
    ...
    print(pose)
    currdir = train_dir + '/' + pose
    poses.append(pose)

    for image in os.listdir(currdir):
        im = io.imread(train_dir + '/' + pose + '/' + image)
        trainim.append(im)
        print(im)
        break

for ax, im, p in zip(grid, trainim, poses):
  ax.imshow(im)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.grid(False)
  ax.set_title(p)

plt.show()

