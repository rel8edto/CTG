import os
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from python_scripts.map_scraper import map_api
from python_scripts.helper import view_random_image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



########### Pre-Processing #################
print("Initializing pre-processing...")
# # Pull data
# emergency = map_api('/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/csv_files/Outpatient-Care-and-Labs.csv',
#                     zoom=18, map_api='google', num_images=5000)

# Walk through data directory and list number of files
for dirpath, dirnames, filenames in os.walk("data"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# # Get the classnames programmatically
# data_dir = pathlib.Path("data/img")
# # Created a list of class_names from the subdirectories
# class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
# class_names = class_names[0:]
# print(class_names)

# # View a random image from the training dataset
# img = view_random_image(target_dir="/Users/nathanieljacques/Rel8ed.to/CTG/prototype/data/img",
#                         target_class="General Freight Trucking")




# ################# Model-Prototype_0: YOLO ######################


# ################ Model-Prototype_0: CNN #########################
# # Set the seed
# tf.random.set_seed(42)

# # Preprocess data (get all of the pixel values between 0 & 1, also called scaling/normalization)
# train_datagen = ImageDataGenerator(rescale=1./255)
# valid_datagen = ImageDataGenerator(rescale=1./255)

# # Setup path to our data directory
# train_dir = "train/Religious Organizations"
# test_dir = "test/Religious Organizations"

# # Import data from directories and turn it into batches
# train_data = train_datagen.flow_from_directory(directory=train_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=42)

# valid_data = valid_datagen.flow_from_directory(directory=test_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=42)

# # Build a CNN model 
# model_1 = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters=10,
#                            kernel_size=3,
#                            activation="relu",
#                            input_shape=(224, 224, 3)),
#     tf.keras.layers.Conv2D(10, 3, activation="relu"),
#     tf.keras.layers.MaxPool2D(pool_size=2,
#                               padding="valid"),

#     tf.keras.layers.Conv2D(10, 3, activation="relu"),
#     tf.keras.layers.Conv2D(10, 3, activation="relu"),
#     tf.keras.layers.MaxPool2D(2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1, activation="sigmoid"),
# ])

# # Compile our model
# model_1.compile(loss="binary_crossentropy",
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=["accuracy"])

# # Fit the model
# history_0 = model_1.fit(train_data,
#                         epochs=5,
#                         steps_per_epoch=len(train_data),
#                         validation_data=valid_data,
#                         validation_steps=len(valid_data))
