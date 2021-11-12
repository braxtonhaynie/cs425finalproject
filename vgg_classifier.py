from keras.callbacks.callbacks import History
from scipy.ndimage.measurements import label
import tensorflow as tf
from keras.utils import to_categorical
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize, shape, std
from skimage import io
from skimage import filters
from skimage import color
from skimage.transform import resize
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

maxImages = 5

imsize = (150, 150)
train_dir = 'dataset/train'
test_dir = 'dataset/test'

labels = {'downdog': 0, 'goddess': 1, 'plank': 2, 'tree': 3, 'warrior2': 4}

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
      y.append(labels.get(pose)) #Appends the pose to keep the same order as the images    
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
      
      X.append(im)
      i += 1
      if i == maxImages:
        break
  return X, y











# for pose in os.listdir(dir):
#   print(pose)
#   currdir = train_dir + '/' + pose
#   i = 0
#   for image in os.listdir(currdir):
#     print(" ({})".format(image))
#     y.append(pose) #Appends the pose to keep the same order as the images    
#     im = io.imread(train_dir + '/' + pose + '/' + image)
#     im = resize(im, imsize)

#     print(shape(im), file=os.sys.stderr)

















train_ds, train_labels = read_images(train_dir)
test_ds, test_labels = read_images(test_dir)
train_ds = np.array(train_ds)
test_ds = np.array(test_ds)
size = (150, 150)

train_ds = tf.image.resize(train_ds, (150, 150))
test_ds = tf.image.resize(test_ds, (150, 150))

train_labels = to_categorical(train_labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

from keras.applications.vgg16 import VGG16, preprocess_input

train_ds = preprocess_input(train_ds)
test_ds = preprocess_input(test_ds)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
base_model.trainable = False

from keras import layers, models


flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation="relu")
dense_layer_2 = layers.Dense(20, activation="relu")
prediction_layer = layers.Dense(5, activation="softmax")


model = models.Sequential(
    [base_model, flatten_layer, dense_layer_1, dense_layer_2, prediction_layer]
)

from keras.callbacks import EarlyStopping

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


es = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
)

history = model.fit(
    train_ds,
    train_labels,
    epochs=20,
    callbacks=[es],
    steps_per_epoch= 32,
)

print(history.history)

results = model.evaluate(test_ds, test_labels, steps=32)
print("test loss, test acc:", results)