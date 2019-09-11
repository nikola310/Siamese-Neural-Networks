import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from tensorflow.keras import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
from omniglot_loader import OmniglotLoader
from siamese_network import SiameseNetwork

if __name__ == "__main__":
    omg = OmniglotLoader(use_transformations=False)

    sn = SiameseNetwork(model_location='C:/Users/Nikola/Documents/Git/siamese_omniglot/models/2019-09-09 18-00-26/model.h5')

    sn.test(omg)