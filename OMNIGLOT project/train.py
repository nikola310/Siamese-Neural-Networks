import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
from omniglot_loader import OmniglotLoader
import os
from datetime import datetime
from siamese_network import SiameseNetwork

'''
    Script for training model without transformations
'''
if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            with tf.device('GPU:1'):
                omg = OmniglotLoader(use_transformations=True)

                sn = SiameseNetwork()

                sn.train(omg, 200)
        except RuntimeError as e:
            print(e)
    else:
        sn.train(omg, 2)