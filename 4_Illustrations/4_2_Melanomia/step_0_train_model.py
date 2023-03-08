###############################################################################
#                         Load required packages
###############################################################################
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import os
from tqdm import tqdm
import sys

from utils_py import preapare_images, CustomDataGen

# For reproducibility
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

###############################################################################
#                       Set global attributes
###############################################################################
EPOCHS = 400
BATCH_SIZE = 128
VAL_SPLIT = 0.1
IMAGE_SHAPE = (256,256,3)
DO_PREPROCESSING = True

# Data files and directories
file_train_csv = "/home/niklas/Downloads/siim-isic-melanoma/train.csv" #/opt/example-data
file_test_csv = "/home/niklas/Downloads/siim-isic-melanoma/test.csv"
dir_train_images = "/home/niklas/Downloads/siim-isic-melanoma/jpeg/train/"
dir_test_images = "/home/niklas/Downloads/siim-isic-melanoma/jpeg/test/"

###############################################################################
#                 Load CSV data and do the preprocessing
###############################################################################
# Load train and test csv files
train_csv = pd.read_csv(file_train_csv).dropna()
test_csv = pd.read_csv(file_test_csv).dropna()

# Do the preprocessing, i.e. load all images -> reshape/rescale -> save as numpy
if DO_PREPROCESSING:
  print("Preprocess train images...")
  preapare_images(dir_train_images, '/data/train/', IMAGE_SHAPE[0:2])
  print("Preprocess test images...")
  preapare_images(dir_test_images, '/data/test/', IMAGE_SHAPE[0:2])
  
###############################################################################
#               Create train and val split and Data Generators
###############################################################################
val_samples = np.random.choice(len(train_csv), int(VAL_SPLIT * len(train_csv)))
train_samples = np.setdiff1d(range(len(train_csv)), val_samples)

# Train data generator
train_data_gen = CustomDataGen(
  df = train_csv.iloc[train_samples, :], 
  batch_size = BATCH_SIZE,
  augment = True,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)

# Validation data generator
val_data_gen = CustomDataGen(
  df = train_csv.iloc[val_samples,:], 
  batch_size = BATCH_SIZE,
  augment = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)


###############################################################################
#                             Define Model
###############################################################################
image_input = keras.Input(shape = IMAGE_SHAPE)
tabular_input = keras.Input(shape = 10)

# Convolutional part
img = keras.layers.Conv2D(32, (8,8), activation = 'relu')(image_input)
img = keras.layers.AvgPool2D((3,3))(img)
img = keras.layers.Conv2D(64, (6,6), activation = 'relu')(img)
img = keras.layers.AvgPool2D((3,3))(img)
img = keras.layers.Conv2D(128, (4,4), activation = 'relu')(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(128, (2,2), activation = 'relu')(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(256, (2,2), activation = 'relu')(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Flatten()(img)
img = keras.layers.Dense(256, activation = 'relu')(img)
img = keras.layers.Dropout(0.4)(img)
img = keras.layers.Dense(128, activation = 'relu')(img)
img = keras.layers.Dropout(0.3)(img)
img = keras.layers.Dense(64, activation = 'relu')(img)

# Tabular part
tab = keras.layers.Dense(256, activation = 'relu')(tabular_input)
tab = keras.layers.Dropout(0.4)(tab)
tab = keras.layers.Dense(128, activation = 'relu')(tab)
tab = keras.layers.Dense(64, activation = 'relu')(tab)

# Combined part
out = keras.layers.Concatenate()([img, tab])
out = keras.layers.Dense(64, activation = 'relu')(out)
out = keras.layers.Dropout(0.4)(out)
out = keras.layers.Dense(32, activation = 'relu')(out)
out = keras.layers.Dropout(0.3)(out)
out = keras.layers.Dense(1, activation = 'sigmoid')(out)

model = keras.Model(inputs = [image_input, tabular_input], outputs = out)

###############################################################################
#                     Compile and train the model
###############################################################################
class_weights = {
  0: len(train_samples) / (2 * sum(train_data_gen.df.target == 0)),
  1: len(train_samples) / (2 * sum(train_data_gen.df.target == 1))
}

def lr_scheduler(epoch, lr):
  if epoch < 10:
    return 0.01
  elif epoch < 100:
    return 0.001
  elif epoch < 200:
    return 0.0005
  elif epoch < 300:
    return 0.0001
  else:
    return lr * tf.math.exp(-0.05)

callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_' + str(IMAGE_SHAPE[0]) + '_' + str(IMAGE_SHAPE[1]),
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True),
  tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
]

model.compile(
  optimizer = "sgd",
  loss = "binary_crossentropy",
  metrics = ["accuracy"]
)

model.fit(
  x = train_data_gen,
  steps_per_epoch = len(train_samples) // BATCH_SIZE,
  validation_data = val_data_gen,
  class_weight = class_weights,
  validation_steps = len(val_samples) // BATCH_SIZE,
  verbose = 1,
  callbacks = callbacks,
  epochs = EPOCHS
)
