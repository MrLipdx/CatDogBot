# coding: utf-8

# # Cat or Dog Image Classifier

# ___
# ## Imports

# Nothing fancy here most of the imports are standard and frequent in ML

import h5py                     #handeling the dataset
import matplotlib.pyplot as plt #viewing nparrays as images
import numpy as np
import os                       #handeling files and folders
import random
import requests
import string
import tensorflow as tf
import tflearn as tfl

#Layers used in the model
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

#used to import images as nparrays
from PIL import Image


# ___
# ## Some constants

# This is to improve modularity and code redability

# In[2]:


MODEL_NAME = "tflearn_AlexNet"

#Folder Constants
#where data is stored
DATA_PATH = "/media/mrlipdx/F88A16BB8A1675FA/Users/MrLipdx/ml"
#where the images from the dataset are
IMAGE_FOLDER = os.path.join(DATA_PATH, "train")
#where to save the HDF5 files
HDF5_FOLDER = os.path.join(DATA_PATH, "hdf5")
#where to save the trainde modle
MODELS_FOLDER = os.path.join(DATA_PATH, "model")
#where the current modle is located
MODEL_FOLDER = os.path.join(MODELS_FOLDER, MODEL_NAME)

#File Path Constants
#Where to load or save the modle to
MODEL_FILE = os.path.join(MODEL_FOLDER, MODEL_NAME)
#text file describing the train dataset
HDF5_TRAIN_INPUT = os.path.join(HDF5_FOLDER, "cats_and_dogs_train.txt")
#train hdf5 file
HDF5_TRAIN = os.path.join(HDF5_FOLDER, "cats_and_dogs_train.hdf5")
#text file describing the test dataset 
HDF5_TEST_INPUT = os.path.join(HDF5_FOLDER, "cats_and_dogs_test.txt")
#test hdf5 file
HDF5_TEST = os.path.join(HDF5_FOLDER, "cats_and_dogs_test.hdf5")

#The ids given to cats and dogs
CLASS_IDS = { "c" : 0, "d" : 1 }

#size of the images in the dataset
IMAGE_SHAPE = (296,299)
#total number of images test + train
TOTAL_IMAGES = len(os.listdir(IMAGE_FOLDER))
#how much test percentage do we want from the total images
TEST_PERCENTAGE = 0.2
TRAIN_PERCENTAGE = 1 - TEST_PERCENTAGE
TEST_SIZE = int(TOTAL_IMAGES * TEST_PERCENTAGE)
TRAIN_SIZE = TOTAL_IMAGES - TEST_SIZE

#True if you want to train a new network
#False if you want to load a previously trainded network
TRAIN = False

#make shure the notebook is consistent after multiple runs
np.random.seed(42)

# ___
# ## Building the model

def buildModel():
    network = input_data(shape=[None, IMAGE_SHAPE[1], IMAGE_SHAPE[0], 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tfl.DNN(network, checkpoint_path = MODEL_FILE,
                   max_checkpoints = 1, tensorboard_verbose = 0)
    return model

# ___
# ## Data

# I'm using a hdf5 database as input on the model. The data is manualy partitioned to a test set and a traning set. The creation of the databases takes a bit of time and at the end uses 26,6 GB so we will save it to save time in the future.

if __name__ == '__main__':
        
    #formats the filenames to the requierd form, "<filepath> <class>\n"
    def filenames_to_input(filenames, directory , class_ids):
        return "".join(['{} {}\n'.format(
                os.path.join(directory, filename), class_ids[filename[0]])
                for filename in filenames])

    if not os.path.exists(HDF5_FOLDER): 
        os.makedirs(HDF5_FOLDER)
        
    if not os.path.isfile(HDF5_TEST) or not os.path.isfile(HDF5_TRAIN):
        print("Missing one or both datasets.")
        print("Creating datasets...")
        images = np.array(os.listdir(IMAGE_FOLDER))
        
        #spliting the images to a train and a test sets
        np.random.shuffle(images)
        test_images = images[:TEST_SIZE]
        train_images = images[TEST_SIZE:]
        
        if not os.path.isfile(HDF5_TEST):
            print("\tCreating test HDF5 dataset...") 
            with open(HDF5_TEST_INPUT, "w") as test_input_file:
                test_input_file.write(filenames_to_input(test_images, IMAGE_FOLDER, CLASS_IDS))

            tfl.data_utils.build_hdf5_image_dataset(HDF5_TEST_INPUT, 
                                                image_shape = IMAGE_SHAPE,
                                                output_path = HDF5_TEST,
                                                categorical_labels = True)
            print("\tDone.\n")

        if not os.path.isfile(HDF5_TRAIN):
            print("\tCreating train HDF5 dataset...")  
            with open(HDF5_TRAIN_INPUT, "w") as train_input_file:
                train_input_file.write(filenames_to_input(train_images, IMAGE_FOLDER, CLASS_IDS))

            tfl.data_utils.build_hdf5_image_dataset(HDF5_TRAIN_INPUT, 
                                                image_shape = IMAGE_SHAPE,
                                                output_path = HDF5_TRAIN,
                                                categorical_labels = True)
            print("\tDone.")
        print("Done.")
    else:
        print("Both datasets present.")
        
    test_dataset = h5py.File(HDF5_TEST, 'r')
    x_test = test_dataset['X']
    y_test = test_dataset['Y']
    train_dataset = h5py.File(HDF5_TRAIN, 'r')
    x_train = train_dataset['X']
    y_train = train_dataset['Y']

    #building the model
    model = buildModel()

    # ___
    # ## Traning
    # Traning the model takes a bit of time in my case i used my Nvidia GTX 1060 and it still took me 2 H or so. I'm not sure why but it kept crashing because me SO was running out of memory and killing the process. If you have anny idea as to why this is happening please let me know.

    #when traning craches i use the following lines to resume from a chekpoint
    #in this case the traning crashed after step 800
    #CHEKPOINT = "-800"
    #model.load(MODEL_FILE + CHEKPOINT, weights_only = True)
    #print("Model Loaded, evaluating ...")
    #print(model.evaluate(x_test, y_test))
    model.fit(x_train, y_train, n_epoch = 20, validation_set = (x_test, y_test),
           snapshot_step = 200, show_metric = True, run_id = MODEL_NAME)
    model.save(MODEL_FILE)

        
    train_dataset.close()

    print("The traning was sucessful!")

