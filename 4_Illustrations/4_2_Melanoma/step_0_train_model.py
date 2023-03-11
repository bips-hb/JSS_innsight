###############################################################################
#                         Load required packages
###############################################################################
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import os
from tqdm import tqdm
import sys

from utils_py import preapare_images, CustomDataGen

# For reproducibility
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Use second GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)
    
print(tf.config.experimental.get_visible_devices('GPU'))

###############################################################################
#                       Set global attributes
###############################################################################
EPOCHS = 400
BATCH_SIZE = 256
VAL_SPLIT = 0.2
IMAGE_SHAPE = (224,224,3)
DO_PREPROCESSING = False # is very time consuming

# Data files and directories
file_train_csv = "/opt/example-data/siim-isic-melanoma/train.csv"
file_test_csv = "/opt/example-data/siim-isic-melanoma/test.csv"
dir_train_images = "/opt/example-data/siim-isic-melanoma/jpeg/train/"
dir_test_images = "/opt/example-data/siim-isic-melanoma/jpeg/test/"

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
# Define bias for last layer
bias = np.log(sum(train_data_gen.df.target == 1) / sum(train_data_gen.df.target == 0))
bias = keras.initializers.Constant(bias)

image_input = keras.Input(shape = IMAGE_SHAPE)
tabular_input = keras.Input(shape = 10)

# Convolutional part
img = keras.layers.Conv2D(32, (3,3), activation = "relu")(image_input)
img = keras.layers.Conv2D(32, (3,3), activation = "relu")(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(64, (3,3), activation = "relu")(img)
img = keras.layers.Conv2D(64, (3,3), activation = "relu")(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(128, (3,3), activation = "relu")(img)
img = keras.layers.Conv2D(128, (3,3), activation = "relu")(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(256, (2,2), activation = "relu")(img)
img = keras.layers.Conv2D(256, (2,2), activation = "relu")(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(512, (2,2), activation = "relu")(img)
img = keras.layers.Conv2D(512, (2,2), activation = "relu")(img)
img = keras.layers.AvgPool2D((2,2))(img)
img = keras.layers.Conv2D(1024, (2,2), activation = "relu")(img)
img = keras.layers.AvgPool2D((3,3))(img)
img = keras.layers.Flatten()(img)
img = keras.layers.Dense(512, activation = 'relu')(img)
img = keras.layers.Dropout(0.4)(img)
img = keras.layers.Dense(256, activation = 'linear')(img)

# Tabular part
tab = keras.layers.Dense(16, activation = 'relu')(tabular_input)
tab = keras.layers.Dropout(0.3)(tab)
tab = keras.layers.Dense(8, activation = 'linear')(tab)

# Combined part
out = keras.layers.Concatenate()([img, tab])
out = keras.layers.Dense(256, activation = 'relu')(out)
out = keras.layers.Dropout(0.4)(out)
out = keras.layers.Dense(64, activation = 'relu')(out)
out = keras.layers.Dropout(0.3)(out)
out = keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = bias)(out)

model = keras.Model(inputs = [image_input, tabular_input], outputs = out)

###############################################################################
#                     Compile and train the model
###############################################################################

# The dataset is highly unbalanced
class_weights = {
  0: len(train_samples) / (2 * sum(train_data_gen.df.target == 0)),
  1: len(train_samples) / (2 * sum(train_data_gen.df.target == 1))
}

# We use the area under the ROC curve as the validation measure
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_' + str(IMAGE_SHAPE[0]) + '_' + str(IMAGE_SHAPE[1]),
    save_weights_only=False,
    monitor='val_auc',
    mode='max',
    save_best_only=True),
  tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_auc", factor = 0.1, patience = 20, 
    min_lr = 1e-6, mode = "max"),
  tf.keras.callbacks.EarlyStopping(
    patience = 60, restore_best_weights = True,
    monitor = "val_auc", mode = "max")

]

# Compile the model
model.compile(
  optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3, momentum = 0.9),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

# Train the model
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

###############################################################################
#                     Evaluate model
###############################################################################

model = keras.models.load_model('checkpoints/model_224_224')
model.compile(
  optimizer = tf.keras.optimizers.Adam(lr=1e-3),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

test_data_gen = CustomDataGen(
  df = train_csv, 
  batch_size = BATCH_SIZE,
  augment = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)
  
result = model.evaluate(test_data_gen, batch_size = BATCH_SIZE)

print(dict(zip(model.metrics_names, result)))
  

