import sys
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage.transform import AffineTransform
from math import pi
from keras.callbacks import TensorBoard
from os import makedirs
from os.path import exists, join
from time import time
import pickle
import tensorflow as tf
import csv
from tensorflow.keras import Model, Sequential
from omniglot_loader import OmniglotLoader
from datetime import datetime

def copy_network(model):
    '''
        Create a copy of basic neural network
    '''

    copy = Sequential()
    copy.add(Conv2D(filters=64, kernel_size=(10, 10), activation='relu',
        input_shape=input_shape, weights=model.layers[2].get_layer('Conv_layer_1').get_weights()))
    copy.add(MaxPool2D())
    copy.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
        weights=model.layers[2].get_layer('Conv_layer_2').get_weights()))
    copy.add(MaxPool2D())
    copy.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu',
        weights=model.layers[2].get_layer('Conv_layer_3').get_weights()))
    copy.add(MaxPool2D())
    copy.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu',
        weights=model.layers[2].get_layer('Conv_layer_4').get_weights()))
    copy.add(Flatten())
    copy.add(Dense(units=4096, activation='sigmoid',
        weights=model.layers[2].get_layer('Dense_layer_1').get_weights()))

    return copy

def write_output_and_metadata(model, omniglot, training, model_type, test_type, log_dir):

    omniglot.set_current_symbol_index(0)
    output_name = 'output_' + model_type + '_' + test_type + '.tsv'
    meta_output = 'metadata_' + model_type + '_' + test_type + '.tsv'

    if not exists(join(log_dir, meta_output)):
        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            f_metadata.write('{}\t{}\n'.format('Label', 'Detail Label'))

    # First true positives and false negatives
    print('Testing true positives and false negatives...')
    while True:
        if omniglot.is_tp_batches_done():
            break

        pairs, _ = omniglot.get_tp_batch(alphabet, training)

        activations = net_copy.predict(pairs[:, 1])
        predictions = model.predict([pairs[:, 0], pairs[:, 1]])

        with open(join(log_dir, output_name), 'a+') as f_output:
            for xid in range(activations.shape[0]):
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(activations[xid])

        metadata = []
        metadata_detailed = []
        
        for i in range(len(predictions)):
            if predictions[i] < 0.25:
                metadata.append('fn')
                metadata_detailed.append('fn_low')
            elif predictions[i] >= 0.25 and predictions[i] < 0.5:
                metadata.append('fn')
                metadata_detailed.append('fn_high')
            elif predictions[i] >= 0.5 and predictions[i] < 0.75:
                metadata.append('tp')
                metadata_detailed.append('tp_low')
            elif predictions[i] >= 0.75:
                metadata.append('tp')
                metadata_detailed.append('tp_high')



        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            for i in range(len(metadata)):
                    f_metadata.write('{}\t{}\n'.format(metadata[i], metadata_detailed[i]))
        
        '''
        output_detailed = 'metadata_' + model_type + '_' + test_type + '_detailed.tsv'

        with open(join(log_dir, output_detailed), 'a+') as f_metadata:
            for meta in metadata_detailed:
                    f_metadata.write('{}\n'.format(meta))
        '''
    print('Testing finished.')

    omg.set_current_symbol_index(0)

    # Then true negatives and false positives
    print('Testing true negatives and false positives...')
    while True:
        if omniglot.is_tn_batches_done():
            break

        pairs, _ = omniglot.get_tn_batch(alphabet, training)

        activations = net_copy.predict(pairs[:, 1])
        predictions = model.predict([pairs[:, 0], pairs[:, 1]])

        #output_name = 'output_' + model_type + '_' + test_type + '.tsv'
        with open(join(log_dir, output_name), 'a+') as f_output:

            for xid in range(activations.shape[0]):
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(activations[xid])

        metadata = []
        metadata_detailed = []
        
        for i in range(len(predictions)):
            if predictions[i] < 0.25:
                metadata.append('tn')
                metadata_detailed.append('tn_low')
            elif predictions[i] >= 0.25 and predictions[i] < 0.5:
                metadata.append('tn')
                metadata_detailed.append('tn_high')
            elif predictions[i] >= 0.5 and predictions[i] < 0.75:
                metadata.append('fp')
                metadata_detailed.append('fp_low')
            elif predictions[i] >= 0.75:
                metadata.append('fp')
                metadata_detailed.append('fp_high')

        #output_name = 'metadata_' + model_type + '_' + test_type + '.tsv'
        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            for i in range(len(metadata)):
                    f_metadata.write('{}\t{}\n'.format(metadata[i], metadata_detailed[i]))

    print('Testing finished.')


if __name__ == "__main__":

    four_labels = False
    num_classes = 20
    use_transformed_model = False
    test_w_transformations = False
    test_wo_transformations = True
    input_shape = (105, 105, 1)
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    projector_dir = 'projector_data' + '/' + run_start_time
    alphabet = 'Kannada'
    model_type = ['model_tf', 'model']
    test_type = ['test_tf', 'test']

    model_w_tf = load_model('./trained_models/w_transform/model.h5')
    model_wo_tf = load_model('./trained_models/wo_transform/model.h5')
    net_copy = copy_network(model_wo_tf)

    # Test model w/o transformations on alphabet w/o transformations
    log_dir = join(projector_dir, alphabet)
    if not exists(log_dir):
        makedirs(log_dir)
    
    # Test three cases:
    # 1) Original Training and Test Data (No Transformations)
    omg = OmniglotLoader(use_transformations=False)
    omg.set_current_alphabet_index(omg.get_alphabet_index(alphabet, False))
    omg.set_training_evaluation_symbols(False)
    write_output_and_metadata(model_wo_tf, omg, False, 'model', 'test', log_dir)
    # 2) Training (No Transformation), Testing (With Transformations)
    omg.set_tn_batches_done(False)
    omg.set_tp_batches_done(False)
    omg.set_use_transformations(True)
    write_output_and_metadata(model_wo_tf, omg, False, 'model', 'test_tf', log_dir)
    # 3) Transformation in both Training and Testing
    omg.set_tn_batches_done(False)
    omg.set_tp_batches_done(False)
    write_output_and_metadata(model_w_tf, omg, False, 'model_tf', 'test_tf', log_dir)