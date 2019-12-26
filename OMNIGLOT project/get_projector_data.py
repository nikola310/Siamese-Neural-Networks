from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
from os import makedirs
from os.path import exists, join
from tensorflow.keras import Sequential
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

def write_output_and_metadata(model, omniglot, training, model_type, test_type, log_dir, network_copy):

    omniglot._current_symbol_index = 0
    output_name = 'output_' + model_type + '_' + test_type + '.tsv'
    meta_output = 'metadata_' + model_type + '_' + test_type + '.tsv'

    if not exists(join(log_dir, meta_output)):
        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            f_metadata.write('{}\t{}\n'.format('Label', 'Label_Detailed'))

    # First true positives and false negatives
    print('Testing true positives and false negatives...')
    while True:
        if omniglot._tp_batches_done:
            break

        pairs, _ = omniglot.get_tp_batch(alphabet, training)
        
        activations = network_copy.predict(pairs[:, 1])
        predictions = model.predict([pairs[:, 0], pairs[:, 1]])
        
        with open(join(log_dir, output_name), 'a+') as f_output:
            for ac in activations[:]:
                out = np.array2string(ac.flatten(), threshold=np.inf, precision=7, max_line_width=np.inf, separator='\t').replace('[', '').replace(']', '')
                f_output.write(out + '\n')

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
        
    print('Testing finished.')

    omg._current_symbol_index = 0

    # Then true negatives and false positives
    print('Testing true negatives and false positives...')
    while True:
        if omniglot._tn_batches_done():
            break

        pairs, _ = omniglot.get_tn_batch(alphabet, training)
        
        activations = network_copy.predict(pairs[:, 1])
        predictions = model.predict([pairs[:, 0], pairs[:, 1]])

        with open(join(log_dir, output_name), 'a+') as f_output:

            for ac in activations[:]:
                out = np.array2string(ac.flatten(), threshold=np.inf, precision=7, max_line_width=np.inf, separator='\t').replace('[', '').replace(']', '')
                f_output.write(out + '\n')

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

        with open(join(log_dir, meta_output), 'a+') as f_metadata:
            for i in range(len(metadata)):
                    f_metadata.write('{}\t{}\n'.format(metadata[i], metadata_detailed[i]))

    print('Testing finished.')


if __name__ == "__main__":

    four_labels = False
    num_classes = 20
    input_shape = (105, 105, 1)
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    projector_dir = 'projector_data' + '/' + run_start_time
    alphabet = 'Kannada'

    model_w_tf = load_model('./models/2019-09-26 11-04-37/model.h5')
    model_wo_tf = load_model('./models/2019-09-25 18-40-12/model.h5')
    net_copy = copy_network(model_wo_tf)

    # Test model w/o transformations on alphabet w/o transformations
    log_dir = join(projector_dir, alphabet)
    if not exists(log_dir):
        makedirs(log_dir)
    
    # Test three cases:
    # 1) Original Training and Test Data (No Transformations)
    omg = OmniglotLoader(use_transformations=False)
    omg._current_alphabet_index = omg.get_alphabet_index(alphabet, False)
    omg.set_training_evaluation_symbols(False)
    write_output_and_metadata(model_wo_tf, omg, False, 'model', 'test', log_dir, net_copy)
    # 2) Training (No Transformation), Testing (With Transformations)
    omg._tn_batches_done = False
    omg._tp_batches_done = False
    omg.use_transformations = True
    write_output_and_metadata(model_wo_tf, omg, False, 'model', 'test_tf', log_dir, net_copy)
    # 3) Transformation in both Training and Testing
    omg._tn_batches_done = False
    omg._tp_batches_done = False
    write_output_and_metadata(model_w_tf, omg, False, 'model_tf', 'test_tf', log_dir, net_copy)
