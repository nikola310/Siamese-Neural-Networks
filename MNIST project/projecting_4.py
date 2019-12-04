import sys
import random
from tensorflow.keras.models import load_model, Model
import numpy as np
import tensorflow.keras.backend as K
from os.path import join
import pickle
import csv
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from train_models import contrastive_loss, create_pairs

np.set_printoptions(threshold=sys.maxsize)

four_labels = False
num_classes = 10
use_transformed_model = False
testing_with_transformations = True
input_shape = (28, 28)
test_cases = [('./data/model_w_tf_te', './test_models/siamese_model_transformations.h5'), ('./data/model_w_tf_te_tf', './test_models/siamese_model_transformations.h5'),
                ('./data/model_wo_tf_te', './test_models/siamese_model.h5'), ('./data/model_wo_tf_te_tf', './test_models/siamese_model.h5')]

def create_subset(pairs, indices):
    ret = []
    
    subset = []
    subset = [indices[3][j] for j in range(len(indices[3]))]
    ret += pairs[subset].tolist()
    return np.asarray(ret)

def write_output_and_labels(digits_location, model_loc):

    model = load_model(model_loc, custom_objects={'contrastive_loss': contrastive_loss})
    # Create copy of snn
    input2 = Input(shape=input_shape)
    x = Flatten()(input2)
    x = Dense(128, activation='relu', weights=model.layers[2].layers[2].get_weights())(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', weights=model.layers[2].layers[4].get_weights())(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', weights=model.layers[2].layers[6].get_weights(), name="features")(x)
    model2 = Model(input2, x)
    
    pairs_tn_fp = np.load(join(digits_location, 'pairs_tn_fp.npy'))

    with open(join(digits_location, 'true_negatives_low.pkl'), 'rb') as f:
        true_negatives_low = pickle.load(f)

    with open(join(digits_location, 'true_negatives_high.pkl'), 'rb') as f:
        true_negatives_high = pickle.load(f)

    with open(join(digits_location, 'false_positives_low.pkl'), 'rb') as f:
        false_positives_low = pickle.load(f)

    with open(join(digits_location, 'false_positives_high.pkl'), 'rb') as f:
        false_positives_high = pickle.load(f)

    pairs_tp_fn = np.load(join(digits_location, 'pairs_tp_fn.npy'))

    with open(join(digits_location, 'true_positives_low.pkl'), 'rb') as f:
        true_positives_low = pickle.load(f)

    with open(join(digits_location, 'true_positives_high.pkl'), 'rb') as f:
        true_positives_high = pickle.load(f)

    with open(join(digits_location, 'false_negatives_low.pkl'), 'rb') as f:
        false_negatives_low = pickle.load(f)

    with open(join(digits_location, 'false_negatives_high.pkl'), 'rb') as f:
        false_negatives_high = pickle.load(f)

    te_pairs_tp_low = create_subset(pairs_tp_fn, true_positives_low)
    te_pairs_tp_high = create_subset(pairs_tp_fn, true_positives_high)
    te_pairs_fn_low = create_subset(pairs_tp_fn, false_negatives_low)
    te_pairs_fn_high = create_subset(pairs_tp_fn, false_negatives_high)

    te_pairs_tn_low = create_subset(pairs_tn_fp, true_negatives_low)
    te_pairs_tn_high = create_subset(pairs_tn_fp, true_negatives_high)
    te_pairs_fp_low = create_subset(pairs_tn_fp, false_positives_low)
    te_pairs_fp_high = create_subset(pairs_tn_fp, false_positives_high)

    activations_right_tp_low = model2.predict(te_pairs_tp_low[:, 1])

    activations_right_tp_high = model2.predict(te_pairs_tp_high[:, 1])

    activations_right_fn_low = model2.predict(te_pairs_fn_low[:, 1])

    activations_right_fn_high = model2.predict(te_pairs_fn_high[:, 1])


    activations_right_tn_low = model2.predict(te_pairs_tn_low[:, 1])

    activations_right_tn_high = model2.predict(te_pairs_tn_high[:, 1])

    activations_right_fp_low = model2.predict(te_pairs_fp_low[:, 1])

    activations_right_fp_high = model2.predict(te_pairs_fp_high[:, 1])

    with open(join(digits_location, 'output.tsv'), 'w', newline='') as f_output:
        
        for xid in range(activations_right_tp_low.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_tp_low[xid])

        for xid in range(activations_right_tp_high.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_tp_high[xid])
        ###############################################################################
        for xid in range(activations_right_fn_low.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_fn_low[xid])

        for xid in range(activations_right_fn_high.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_fn_high[xid])
        ###############################################################################
        for xid in range(activations_right_tn_low.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_tn_low[xid])

        for xid in range(activations_right_tn_high.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_tn_high[xid])
        ###############################################################################
        for xid in range(activations_right_fp_low.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_fp_low[xid])

        for xid in range(activations_right_fp_high.shape[0]):
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activations_right_fp_high[xid])


    labels = ["tp"] * (len(activations_right_tp_low) + len(activations_right_tp_high)) + ["fn"] * (len(activations_right_fn_low) + len(activations_right_fn_high)) + ["tn"] * (len(activations_right_tn_low) + len(activations_right_tn_high))  + ["fp"] * (len(activations_right_fp_low) + len(activations_right_fp_high))
    labels_detailed = ["tp_low"] * len(activations_right_tp_low) + ["tp_high"] * len(activations_right_tp_high) + ["fn_low"] * len(activations_right_fn_low) + ["fn_high"] * len(activations_right_fn_high) + ["tn_low"] * len(activations_right_tn_low) + ["tn_high"] * len(activations_right_tn_high)  + ["fp_low"] * len(activations_right_fp_low) + ["fp_high"] * len(activations_right_fp_high)

    with open(join(digits_location, 'metadata.tsv'), 'w') as handle:
        handle.write('{}\t{}\n'.format('Label', 'Label_Detailed'))
        i = 0
        for i in range(len(labels)):
                handle.write('{}\t{}\n'.format(labels[i], labels_detailed[i]))
                i += 1

if __name__ == "__main__":

    for case in test_cases:
        write_output_and_labels(case[0], case[1])