import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import tensorflow_datasets as tfds

print(tf.__version__)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname="flower_photos", untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob("*/*.jpg")))

roses = list(data_dir.glob("roses/*"))
PIL.Image.open(str(roses[0])).show()
PIL.Image.open(str(roses[1])).show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

