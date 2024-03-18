# Malaria Dataset 100% data (brilliant scores) -> manual data picking
# https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import random
import os

train_folder_path = 'Machine Learning 3/cell_images/Train'
test_folder_path = 'Machine Learning 3/cell_images/Test'

train_folder = os.listdir(train_folder_path)
test_folder = os.listdir(test_folder_path)

train_img_folders = []
test_img_folders = []

for each in train_folder:
    list_ = []
    for file in os.listdir(f'{train_folder_path}/{each}'):
        list_.append(f'{train_folder_path}/{each}/{file}')
    train_img_folders.append(list_)

for each in test_folder:
    list_ = []
    for file in os.listdir(f'{test_folder_path}/{each}'):
        list_.append(f'{test_folder_path}/{each}/{file}')
    test_img_folders.append(list_)

# create y labels
train_labels = []
j = 0
for each in train_img_folders:
    for i in range(len(each)):
        train_labels.append(j)
    j += 1

test_labels = []
j = 0
for each in test_img_folders:
    for i in range(len(each)):
        test_labels.append(j)
    j += 1


# flatten img paths
train_imgs_paths = np.array(train_img_folders).flatten()
test_imgs_paths = np.array(test_img_folders).flatten()

# preprocess imgs
def preprocess(img, label):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, size=[224, 224])
    return img, label

def create_data_batches(training=False, testing=False, X=None, y=None):
    if training:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(preprocess).shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size=32)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(preprocess)
        dataset = dataset.batch(batch_size=32)
    return dataset

train_data = create_data_batches(training=True, X=train_imgs_paths, y=train_labels)
test_data = create_data_batches(testing=True, X=test_imgs_paths, y=test_labels)



# modelling
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3, 
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3, 
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3, 
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3, 
        activation='relu'
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.fit(
    x=train_data,
    batch_size=32,
    epochs=10,
    shuffle=True,
    verbose=2
)
