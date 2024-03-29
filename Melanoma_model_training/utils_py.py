import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from tqdm import tqdm
import os
import math

CHANNEL_MEANS = np.array((0.80612123, 0.62106454, 0.591202)).reshape((1,1,3))


def get_base_model(img_shape):
  
  # Define the model
  image_input = keras.Input(shape = img_shape)
  
  block1_in = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(image_input)
  img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(block1_in)
  img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(img)
  block1_in = keras.layers.Add()([block1_in, img])
  img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(block1_in)
  img = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Add()([block1_in, img])
  img = keras.layers.AvgPool2D((3,3))(img)
  
  block2_in = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(block2_in)
  img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(img)
  block2_in = keras.layers.Add()([block2_in, img])
  img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(block2_in)
  img = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Add()([block2_in, img])
  img = keras.layers.AvgPool2D((3,3))(img)
  
  block3_in = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block3_in)
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(img)
  block3_in = keras.layers.Add()([block3_in, img])
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block3_in)
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Add()([block3_in, img])
  img = keras.layers.AvgPool2D((3,3))(img)
  
  block4_in = keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block4_in)
  img = keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu")(img)
  block4_in = keras.layers.Add()([block4_in, img])
  img = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(block4_in)
  img = keras.layers.Conv2D(256, (3,3), padding = "same", activation = "relu")(img)
  img = keras.layers.Add()([block4_in, img])
  img = keras.layers.AvgPool2D((3,3))(img)
  
  out = keras.layers.Flatten()(img)
  out = keras.layers.Dense(512, activation = 'relu', name = "base_model_1")(out)
  out = keras.layers.Dropout(0.3, name = "base_model_2")(out)
  out = keras.layers.Dense(256, activation = 'linear', name = "base_model_end")(out)
  out = keras.layers.Dropout(0.3)(out)
  out = keras.layers.Dense(1, activation = 'sigmoid')(out)
  
  model = keras.Model(inputs = image_input, outputs = out)
  
  return model

def get_image(img_path, size):
  image = tf.keras.preprocessing.image.load_img(img_path)
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = tf.image.resize(image,size).numpy()
    
  return image / 255.0 - CHANNEL_MEANS
  
  
def preapare_images(source, dest, shape):
  # Create destination folder
  dest = os.getcwd() + dest
  if not os.path.exists(dest):
    os.makedirs(dest)
    
  for img_name in tqdm(os.listdir(source)):
    img_arr = get_image(source + img_name, shape)
    np.save(dest + img_name.split('.')[0] + '.npy', img_arr)
    
    
    
def encode_df(df, img_source):
  if not 'target' in df.keys():
    df['target'] = pd.NA
    
  encoded_df = pd.DataFrame({
    'image_name': img_source + df['image_name'] + '.npy',
    'sex_male': (df['sex'] == 'male').astype(float),
    'sex_female': (df['sex'] == 'female').astype(float),
    'age': df['age_approx'] / 90.0 - 0.5430002,
    'loc_head_neck': (df['anatom_site_general_challenge'] == 'head/neck').astype(float),
    'loc_torso': (df['anatom_site_general_challenge'] == 'torso').astype(float),
    'loc_upper_extrem': (df['anatom_site_general_challenge'] == 'upper extremity').astype(float),
    'loc_lower_extrem': (df['anatom_site_general_challenge'] == 'lower extremity').astype(float),
    'loc_palms_soles': (df['anatom_site_general_challenge'] == 'palms/soles').astype(float),
    'loc_oral_genital': (df['anatom_site_general_challenge'] == 'oral/genital').astype(float),
    'loc_missing': (df['anatom_site_general_challenge'].isna()).astype(float),
    'target': df['target']
  })
  
  return encoded_df


class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df,
                 batch_size,
                 img_source,
                 has_target=True,
                 augment = True,
                 input_size=(32, 32, 3),
                 shuffle=True,
                 omit_tab_data=False):
        
        self.df = encode_df(df.copy(), img_source)
        self.batch_size = batch_size
        self.input_size = input_size
        self.augment = augment
        self.shuffle = shuffle
        self.omit_tab_data = omit_tab_data
        self.n = len(self.df)
        self.crop_size = [int(input_size[0] * 1.2), int(input_size[1] * 1.2)]
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_data(self, batches):
        image_batch = np.stack([np.load(img_path) for img_path in batches['image_name']])
        if self.augment:
          image_batch = tf.image.random_flip_left_right(image_batch)
          image_batch = tf.image.random_flip_up_down(image_batch)
          image_batch = tf.image.resize(image_batch, self.crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          image_batch = tf.image.random_crop(image_batch, (image_batch.shape[0],) + self.input_size)
          image_batch = tf.image.random_brightness(image_batch, 0.1)
          image_batch = tf.image.random_contrast(image_batch, lower = 0.9, upper = 1.1)
          image_batch = tf.image.random_hue(image_batch, 0.1)
          
        y_batch = batches['target'].to_numpy()
        
        if self.omit_tab_data:
          return image_batch, y_batch
        else:
          tab_batch = batches.iloc[:, 1:11].to_numpy()
          
          return [image_batch, tab_batch], y_batch
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.n)
        
        batches = self.df[start:end]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return math.ceil(self.n / self.batch_size)
  
def drop_nan(df):
  return df[df["sex"].notna() & df["age_approx"].notna()]
