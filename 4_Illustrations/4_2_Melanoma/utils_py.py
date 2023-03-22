import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math

CHANNEL_MEANS = np.array((0.80612123, 0.62106454, 0.591202)).reshape((1,1,3))

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
                 shuffle=True):
        
        self.df = encode_df(df.copy(), img_source)
        self.batch_size = batch_size
        self.input_size = input_size
        self.augment = augment
        self.shuffle = shuffle
        self.n = len(self.df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_data(self, batches):
        image_batch = np.stack([np.load(img_path) for img_path in batches['image_name']])
        if self.augment:
          image_batch = tf.image.random_flip_left_right(image_batch)
          image_batch = tf.image.random_flip_up_down(image_batch)
          image_batch = tf.image.random_brightness(image_batch, 0.2)
          image_batch = tf.image.random_contrast(image_batch, lower = 0.8, upper = 1.2)
          image_batch = tf.image.random_hue(image_batch, 0.1)
        tab_batch = batches.iloc[:, 1:11].to_numpy()
        y_batch = batches['target'].to_numpy()

        return [image_batch, tab_batch], y_batch
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.n)
        
        batches = self.df[start:end]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return math.ceil(self.n / self.batch_size)


def get_predictions(model_path, batch_size, data_csv_path, image_shape = (224, 224, 3)):
  # Load data csv
  data_csv = drop_nan(pd.read_csv(data_csv_path))
  # Create data generator
  data_gen = CustomDataGen(
    df = data_csv, 
    batch_size = int(batch_size),
    augment = False,
    img_source = os.getcwd() + '/4_Illustrations/4_2_Melanoma/data/train/', 
    input_size = image_shape,
    shuffle = False)
  
  # load model
  model = tf.keras.models.load_model(model_path)
  
  preds = model.predict(data_gen, batch_size = int(batch_size))
  
  return preds
  
def drop_nan(df):
  return df[df["sex"].notna() & df["age_approx"].notna()]
