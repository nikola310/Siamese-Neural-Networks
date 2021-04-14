import csv
import numpy as np
import sys
import tensorflow.keras.backend as K

from datetime import datetime
from omniglot_loader import OmniglotLoader
from os import makedirs
from os.path import exists, join
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Denses
from tensorflow.keras.models import load_model

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

    omniglot._current_symbol_index = 0
    output_name = 'output_' + model_type + '_' + test_type + '.tsv'
    meta_output = 'metadata_' + model_type + '_' + test_type + '.tsv'

    if not exists(join(log_dir, meta_output)):
        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            f_metadata.write('{}\t{}\n'.format('Label', 'Detail Label'))

    # First true positives and false negatives
    print('Testing true positives and false negatives...')
    while True:
        if omniglot._tp_batches_done:
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

    omg._current_symbol_index = 0

    # Then true negatives and false positives
    print('Testing true negatives and false positives...')
    while True:
        if omniglot._tn_batches_done:
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
    np.set_printoptions(threshold=sys.maxsize)
    four_labels = False
    num_classes = 20
    use_transformed_model = False
    test_w_transformations = False
    test_wo_transformations = True
    input_shape = (105, 105, 1)
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    projector_dir = 'projector_data' + '/' + run_start_time
    alphabet = 'Kannada'

    model = load_model('./trained_models/wo_transform/model.h5')
    net_copy = copy_network(model)

    # Test model w/o transformations on alphabet w/o transformations
    log_dir = join(projector_dir, alphabet)
    if not exists(log_dir):
        makedirs(log_dir)
    OutFunc = K.function([model.layers[2].get_layer('Conv_layer_1').input, [model.layers[0].input, model.layers[1].input]], [model.layers[2].get_layer('Dense_layer_1').output[1], model.layers[4].output])
    omg = OmniglotLoader(use_transformations=False)
    images, labels = omg.get_training_batch()
    print(model.summary())
    print(images[0, 0].dtype)
    print(model.input)
    K.clear_session()
    out_val = OutFunc([[images[:, 0], images[:, 1]], [images[:, 0], images[:, 1]]])
    print(out_val[1])
    #print(out_val)
    print(len(out_val[0][0])) # activations
    print(out_val[1][0]) # predictions
    print(len(out_val[0][1])) # activations
    print(out_val[1][1]) # predictions
    print(model.layers[2].get_layer('Dense_layer_1').output[0])
