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

from utils_py import preapare_images, CustomDataGen, drop_nan

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
VAL_SPLIT = 0.15
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
  shuffle = True,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)

# Validation data generator
val_data_gen = CustomDataGen(
  df = train_csv.iloc[val_samples,:], 
  batch_size = BATCH_SIZE,
  augment = False,
  shuffle = False,
  img_source = os.getcwd() + '/data/train/', 
  input_size = IMAGE_SHAPE)
  
print("Number of Training Images = ", train_data_gen.n, " (", sum(train_data_gen.df.target == 1), ")")
print("Number of Validation Images = ", val_data_gen.n, " (", sum(val_data_gen.df.target == 1), ")")

###############################################################################
#                             Define Model
###############################################################################
# Define bias for last layer
bias = np.log(sum(train_data_gen.df.target == 1) / sum(train_data_gen.df.target == 0))
bias = keras.initializers.Constant(bias)

image_input = keras.Input(shape = IMAGE_SHAPE)
tabular_input = keras.Input(shape = 10)

block1_in = keras.layers.Conv2D(16, (6,6), padding = "same", activation = "relu", use_bias = False)(image_input)
img = keras.layers.Conv2D(16, (6,6), padding = "same", activation = "relu")(block1_in)
img = keras.layers.Conv2D(16, (6,6), padding = "same", activation = "relu")(img)
block1_in = keras.layers.Add()([block1_in, img])
img = keras.layers.Conv2D(16, (6,6), padding = "same", activation = "relu")(block1_in)
img = keras.layers.Conv2D(16, (6,6), padding = "same", activation = "relu")(img)
img = keras.layers.Add()([block1_in, img])
img = keras.layers.AvgPool2D((3,3))(img)

block2_in = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu", use_bias = False)(img)
img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(block2_in)
img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(img)
block2_in = keras.layers.Add()([block2_in, img])
img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(block2_in)
img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(img)
img = keras.layers.Add()([block2_in, img])
img = keras.layers.AvgPool2D((3,3))(img)

block3_in = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu", use_bias = False)(img)
img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(block3_in)
img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(img)
block3_in = keras.layers.Add()([block3_in, img])
img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(block3_in)
img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(img)
img = keras.layers.Add()([block3_in, img])
img = keras.layers.AvgPool2D((3,3))(img)

block4_in = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu", use_bias = False)(img)
img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block4_in)
img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(img)
block4_in = keras.layers.Add()([block4_in, img])
img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block4_in)
img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(img)
img = keras.layers.Add()([block4_in, img])
img = keras.layers.AvgPool2D((4,4))(img)

out = keras.layers.Flatten()(img)
out = keras.layers.Dense(256, activation = 'relu')(out)
out = keras.layers.Dropout(0.2)(out)
image_model_out = keras.layers.Dense(128, activation = 'linear')(out)

# Tabular part
tab = keras.layers.Dense(64, activation = 'relu', name = "tab_layer_1")(tabular_input)
tab = keras.layers.Dropout(0.2, name = "tab_layer_2")(tab)
tab = keras.layers.Dense(32, activation = 'linear', name = "tab_layer_3")(tab)
tab = keras.layers.Dropout(0.2, name = "tab_layer_4")(tab)
tab_model_out = keras.layers.Dense(16, activation = 'linear', name = "tab_layer_5")(tab)

# Combined part
out = keras.layers.Concatenate()([image_model_out, tab_model_out])
out = keras.layers.Dense(256, activation = 'relu')(out)
out = keras.layers.Dropout(0.2)(out)
out = keras.layers.Dense(1, activation = 'sigmoid', bias_initializer = bias)(out)

model = keras.Model(inputs = [image_input, tabular_input], outputs = out)

# Set the tabular model as non-trainable
for layer in model.layers:
  if layer.name.startswith("tab_layer"):
    layer.trainable = False

print(model.summary())

###############################################################################
#                     Compile and train the model
###############################################################################

# The dataset is highly unbalanced
class_weights = {
  0: len(train_samples) / (2 * sum(train_data_gen.df.target == 0)),
  1: len(train_samples) / (2 * sum(train_data_gen.df.target == 1))
}

# Warm-up training for the CNN
# Compile the model
model.compile(
  optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
  loss = "binary_crossentropy",
  metrics = [keras.metrics.AUC(name = "auc"), "accuracy"]
)

# Train the model
model.fit(
  x = train_data_gen,
  steps_per_epoch = len(train_samples) // BATCH_SIZE,
  class_weight = class_weights,
  verbose = 1,
  epochs = 100
)

# We use the area under the ROC curve as the validation measure
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_' + str(IMAGE_SHAPE[0]) + '_' + str(IMAGE_SHAPE[1]),
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True),
  tf.keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss", factor = 0.1, patience = 20, 
    min_lr = 1e-6, mode = "min"),
  tf.keras.callbacks.EarlyStopping(
    patience = 60, restore_best_weights = True,
    monitor = "val_loss", mode = "min")
]

# Set the tabular model as non-trainable
for layer in model.layers:
  if layer.name.startswith("tab_layer"):
    layer.trainable = True

model.compile(
  optimizer = keras.optimizers.Adam(learning_rate = 1e-4),
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
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
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
  

