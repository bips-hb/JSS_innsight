###############################################################################
#                         Load required packages
###############################################################################
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tensorflow.keras as keras
import os
from tqdm import tqdm
import sys

from utils_py import preapare_images, CustomDataGen, drop_nan, get_base_model

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
EPOCHS = 2
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
train_csv = drop_nan(pd.read_csv(file_train_csv))
test_csv = drop_nan(pd.read_csv(file_test_csv))

# Do the preprocessing, i.e. load all images -> reshape/rescale -> save as numpy
if DO_PREPROCESSING:
  print("Preprocess train images...")
  preapare_images(dir_train_images, '/data/train/', IMAGE_SHAPE[0:2])
  print("Preprocess test images...")
  preapare_images(dir_test_images, '/data/test/', IMAGE_SHAPE[0:2])
  
val_samples = np.random.choice(len(train_csv), int(VAL_SPLIT * len(train_csv)))
train_samples = np.setdiff1d(range(len(train_csv)), val_samples)

###############################################################################
#                  Warm up: Training without tabular data
###############################################################################

# Train data generator for the base model
train_data_gen = CustomDataGen(
  df = train_csv.iloc[train_samples, :], 
  batch_size = BATCH_SIZE,
  augment = True,
  shuffle = True,
  omit_tab_data = True,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)
  
# Valid data generator for the base model
val_data_gen = CustomDataGen(
  df = train_csv.iloc[val_samples, :], 
  batch_size = BATCH_SIZE,
  augment = False,
  shuffle = False,
  omit_tab_data = True,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)

# The dataset is highly unbalanced
class_weights = {
  0: len(train_samples) / (2 * sum(train_data_gen.df.target == 0)),
  1: len(train_samples) / (2 * sum(train_data_gen.df.target == 1))
}

# get base model
base_model = get_base_model(IMAGE_SHAPE)
base_model.compile(
  optimizer = keras.optimizers.SGD(learning_rate = 1e-2, momentum = 0.9),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

print(base_model.summary())

callbacks = [
  tf.keras.callbacks.ReduceLROnPlateau(
    verbose = 1,
    monitor = "val_auc", factor = 0.1, patience = 30, 
    min_lr = 1e-6, mode = "max"),
  tf.keras.callbacks.EarlyStopping(
    verbose = 1,
    patience = 60, restore_best_weights = True,
    monitor = "val_auc", mode = "max")
]

# Train the model
base_model.fit(
  x = train_data_gen,
  validation_data = val_data_gen,
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1,
  epochs = 2
)

#  Save the base model
base_model.save('checkpoints/base_model_' + str(IMAGE_SHAPE[0]) + '_' + str(IMAGE_SHAPE[1]))

###############################################################################
#               Create train and val split and Data Generators
###############################################################################

# Train data generator
train_data_gen = CustomDataGen(
  df = train_csv.iloc[train_samples, :], 
  batch_size = BATCH_SIZE,
  augment = True,
  shuffle = True,
  omit_tab_data = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)

# Validation data generator
val_data_gen = CustomDataGen(
  df = train_csv.iloc[val_samples,:], 
  batch_size = BATCH_SIZE,
  augment = False,
  shuffle = False,
  omit_tab_data = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)

###############################################################################
#             Combine base model with the tabular model
###############################################################################

base_model = tf.keras.models.load_model('checkpoints/base_model_' + str(IMAGE_SHAPE[0]) + '_' + str(IMAGE_SHAPE[1]))

# Get relevant inputs and oputputs of the base model
image_input = base_model.input
base_model_out = base_model.get_layer("base_model_end").output

# Define bias for last layer
bias = np.log(sum(train_data_gen.df.target == 1) / sum(train_data_gen.df.target == 0))
bias = keras.initializers.Constant(bias)

# Tabular part
tabular_input = keras.Input(shape = 10, name = "input_2")
tab = keras.layers.Dense(32, activation = 'relu')(tabular_input)
tab = keras.layers.Dropout(0.2)(tab)
tab = keras.layers.Dense(16, activation = 'linear')(tab)
tab = keras.layers.Dropout(0.2)(tab)
tab_model_out = keras.layers.Dense(8, activation = 'linear')(tab)

# Combined part
out = keras.layers.Concatenate()([base_model_out, tab_model_out])
out = keras.layers.Dense(256, activation = 'relu')(out)
out = keras.layers.Dropout(0.3)(out)
out = keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = bias)(out)

model = keras.Model(inputs = [image_input, tabular_input], outputs = out)
print(model.summary())


###############################################################################
#                     Compile and train the model
###############################################################################

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
    patience = 40, restore_best_weights = True,
    monitor = "val_auc", mode = "max")
]

model.compile(
  optimizer = keras.optimizers.SGD(learning_rate = 1e-2, momentum = 0.9),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

# Train the model
model.fit(
  x = train_data_gen,
  validation_data = val_data_gen,
  class_weight = class_weights,
  verbose = 1,
  callbacks = callbacks,
  epochs = EPOCHS
)

###############################################################################
#                     Evaluate model
###############################################################################

model = keras.models.load_model('checkpoints/model_224_224')
model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

# Results on the validation data
result = model.evaluate(val_data_gen, batch_size = BATCH_SIZE)

print("Results on the validation data:")
print(dict(zip(model.metrics_names, result)))

print("\n")
print("Results on the whole dataset:")
test_data_gen = CustomDataGen(
  df = train_csv, 
  batch_size = BATCH_SIZE,
  augment = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)
  
result = model.evaluate(test_data_gen, batch_size = BATCH_SIZE)

print(dict(zip(model.metrics_names, result)))
  

