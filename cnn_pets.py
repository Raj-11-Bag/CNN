# -*- coding: utf-8 -*-
"""CNN Pets.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1n7GFwZFQKTs9dGqxXA7avuPMnXHGQkvr
"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
dataset = tfds.load('oxford_iiit_pet', split='train', as_supervised=False)

# Preprocess the data
def preprocess(features):
    image = tf.image.resize(features['image'], [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    label = features['label']
    return image, label

# Apply the preprocessing function
dataset = dataset.map(preprocess)

# Batch and shuffle the data
dataset = dataset.shuffle(1000).batch(32)

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(37, activation='softmax')  # There are 37 classes in the Oxford-IIIT Pet dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10)

import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset
dataset = tfds.load('oxford_iiit_pet', split='train', as_supervised=True)

# Split the dataset into train and test sets
num_examples = len(dataset)
train_size = int(0.7 * num_examples)
test_size = num_examples - train_size

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Preprocess the data
def preprocess(image, label):
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply the preprocessing function
train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Batch and shuffle the data
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(37, activation='softmax')  # There are 37 classes in the Oxford-IIIT Pet dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

test_loss,test_acc = model.evaluate(test_dataset)

model.summary()