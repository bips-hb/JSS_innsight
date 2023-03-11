
# `innsight` and the Melanoma Dataset

## Step 0: Train a Neural Network

### Create Conda environment

To avoid screwing things up on the machine, we create a Conda environment
with the following requirements:

- Python==3.8.13
- numpy==1.23.4
- TensorFlow==2.9.3
- Keras==2.9.0
- Cuda==11.2.2 (for GPU usage)
- cudNN==8.1.0 (for GPU usage)

For a correct installation of the environment we refer to the official site
of [TensorFlow](https://www.tensorflow.org/install/pip).

### Train the model

The melanoma dataset consists of over $33.000$ training images, partly in
still very high resolution. Since these have to be repeatedly loaded into
memory during training and scaled to a unit size, this is done as a
preprocessing step before the actual training. Therefore, change the following
lines in the global attributes section in the file `step_0_train_model.py` file,
accordingly:

```python
EPOCHS = 400
BATCH_SIZE = 256
VAL_SPLIT = 0.1
IMAGE_SHAPE = (224,224,3)
DO_PREPROCESSING = True

# Data files and directories
file_train_csv = "/opt/example-data/siim-isic-melanoma/train.csv"
file_test_csv = "/opt/example-data/siim-isic-melanoma/test.csv"
dir_train_images = "/opt/example-data/siim-isic-melanoma/jpeg/train/"
dir_test_images = "/opt/example-data/siim-isic-melanoma/jpeg/test/"
```

- `EPOCHS`: This integer value is used to set the number of epochs for the
training. Furthermore, 'Adam' is used as the optimizer and the callback
'EarlyStopping' ensures that only the model with the best validation
loss will be saved.
- `BATCH_SIZE`: Make sure that the entire batch of data fits into the memory.
- `VAL_SPLIT`: Set the fraction of the training data to be used exclusively as
validation data. These are then randomly taken from all training data.
- `IMAGE_SHAPE`: This triple defines the uniform shape of all input images
for the model, i.e. each input image must be scaled to this size before it can
be fed into the model. In the preprocessing step all images are loaded, scaled
to this shape and stored as a numpy array under `~/data/train/` or
`~/data/test/` with the same file name.
- `DO_PREPROCESSING`: This logical value decides whether the preprocessing step
will be executed again. Once it must be set to `True` for each change of
`IMAGE_SHAPE`, otherwise no data exist or they have the wrong shape.
- Subsequently, the paths of the melanoma data set must be specified. The
path to the respective csv files and the respective paths to the directories
with the jpeg images for the training and test data are required.


Then, the model can be trained on the melanoma dataset by calling the Python
file `step_0_train_model.py` in the created Conda environment:

```bash
python step_0_train_model.py
```

The model with the best validation loss is then saved in the folder
`~/checkpoints/` folder under the name `model_{IMAGE_SHAPE[0]}_{IMAGE_SHAPE[1]}`.


## Step 1: Create the `Converter` object














