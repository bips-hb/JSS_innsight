
# Training a model on the melanoma dataset

## Preprocessing

### Create Conda environment

To avoid screwing things up on the machine, we create a Conda environment
with the following requirements:

- Python==3.9
- numpy==1.23
- pandas==1.5.3
- TensorFlow==2.9
- Keras==2.9.0
- Cuda==11.2.2 (for GPU usage)
- cudNN==8.1.0 (for GPU usage)

For a correct installation of the environment we refer to the official site
of [TensorFlow](https://www.tensorflow.org/install/pip).

### Gloabl attributes

The entire training procedure is carried out using the Python script 
`step_0_train_model.py`. It starts with defining global attributes, such as 
batch size, input image size, and the proportion of training data used as 
validation data. Subsequently, the paths of the melanoma data set must be 
specified. The path to the respective csv files and the respective paths to 
the directories with the jpeg images for the training and test data are required.

```python
EPOCHS = 400
BATCH_SIZE = 256
VAL_SPLIT = 0.2
IMAGE_SHAPE = (224,224,3)
DO_PREPROCESSING = True

# Data files and directories
file_train_csv = "/opt/example-data/siim-isic-melanoma/train.csv"
file_test_csv = "/opt/example-data/siim-isic-melanoma/test.csv"
dir_train_images = "/opt/example-data/siim-isic-melanoma/jpeg/train/"
dir_test_images = "/opt/example-data/siim-isic-melanoma/jpeg/test/"
```

If everything is installed correctly and the global attributes are set 
appropriately, the following terminal command will execute all the subsequent 
steps and train a model after activating the conda environment:

```bash
python step_0_train_model.py
```

### Data preprocessing

The melanoma dataset consists of over $33.000$ training images, partly in
still very high resolution. Since these have to be repeatedly loaded into
memory during training, scaled to a unit size and normalized, this is done as a
preprocessing step before the actual training. The corresponding `numpy` files 
are stored in the folders `data/train/` for the train data and `data/test/` for
the unlabeled test data.

```python
# Load train and test csv files
train_csv = drop_nan(pd.read_csv(file_train_csv))
test_csv = drop_nan(pd.read_csv(file_test_csv))

# Do the preprocessing, i.e. load all images -> reshape/rescale -> save as numpy
if DO_PREPROCESSING:
  print("Preprocess train images...")
  preapare_images(dir_train_images, '/data/train/', IMAGE_SHAPE[0:2])
  print("Preprocess test images...")
  preapare_images(dir_test_images, '/data/test/', IMAGE_SHAPE[0:2])
```

## Warm up training

As we are training a deep model based on residual layers from scratch and 
the image model is much more complex and parameter-rich than the patient 
information model, we initially train only the image model for $300$ epochs on 
the image data. To achieve this, we utilize the optimizer Stochastic Gradient 
Descent (SGD) with a momentum of $0.9$. We multiply the initial learning rate 
of $0.01$ by $0.1$ after every $30$ epochs and stop training when no 
improvement in validation AUC has occurred for longer than $60$ epochs. 
The primary purpose of this step is to obtain a well-fitted model for the 
images before training it along with the tabular model.

Then, the model can be trained on the melanoma dataset by calling the Python
file `step_0_train_model.py` in the created Conda environment:

```bash
python step_0_train_model.py
```

The model with the best validation AUC is then saved in the folder
`~/checkpoints/` folder under the name `base_model_{IMAGE_SHAPE[0]}_{IMAGE_SHAPE[1]}`.

## Model training

Next, the image model is extended with the dense layers of tabular input and 
the combined output layers after concatenation. This results in a model with a 
total of $3\,069\,113$ parameters. This model is also trained with the 
optimizer SGD with an initial learning rate of $0.1$ and momentum of $0.9$. 
Furthermore, the learning rate is multiplied by a factor of $0.1$ after $20$, 
and the training is stopped after $40$ epochs of no validation AUC improvement.
Using this approach, we achieved a validation AUC of $87.71\%$ and
accuracy of $84.19\%$ after $21$ epochs.

## Notes

* Since this dataset is a highly imbalanced dataset, a weighted binary 
crossentropy was used as the loss function. The weights were calculated as follows:
```python
class_weights = {
  0: len(train_samples) / (2 * sum(train_data_gen.df.target == 0)),
  1: len(train_samples) / (2 * sum(train_data_gen.df.target == 1))
}
```

* In addition, augmentation of the image data was applied to both the warm up 
training and the main training procedure:
```python
image_batch = tf.image.random_flip_left_right(image_batch)
image_batch = tf.image.random_flip_up_down(image_batch)
image_batch = tf.image.resize(image_batch, (268,268), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
image_batch = tf.image.random_crop(image_batch, (image_batch.shape[0],) + self.input_size)
image_batch = tf.image.random_brightness(image_batch, 0.1)
image_batch = tf.image.random_contrast(image_batch, lower = 0.9, upper = 1.1)
image_batch = tf.image.random_hue(image_batch, 0.1)
```












