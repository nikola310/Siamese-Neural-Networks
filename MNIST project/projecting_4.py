from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from train_models import contrastive_loss
import numpy as np
from os.path import join
import pickle
import csv
from projecting_4 import create_subset

test_cases = [('./data/model_w_tf_te', './test_models/siamese_model_transformations.h5'), ('./data/model_w_tf_te_tf', './test_models/siamese_model_transformations.h5'),
                ('./data/model_wo_tf_te', './test_models/siamese_model.h5'), ('./data/model_wo_tf_te_tf', './test_models/siamese_model.h5')]  

def load_pairs(digits_location):
    pairs_tn_fp = np.load(join(digits_location, 'pairs_tn_fp.npy'))
    pairs_tp_fn = np.load(join(digits_location, 'pairs_tp_fn.npy'))
    
    return pairs_tn_fp,  pairs_tp_fn
    
def load_test_data(digits_location):
    with open(join(digits_location, 'true_negatives_low.pkl'), 'rb') as f:
        true_negatives_low = pickle.load(f)

    with open(join(digits_location, 'true_negatives_high.pkl'), 'rb') as f:
        true_negatives_high = pickle.load(f)

    with open(join(digits_location, 'false_positives_low.pkl'), 'rb') as f:
        false_positives_low = pickle.load(f)

    with open(join(digits_location, 'false_positives_high.pkl'), 'rb') as f:
        false_positives_high = pickle.load(f)

    with open(join(digits_location, 'true_positives_low.pkl'), 'rb') as f:
        true_positives_low = pickle.load(f)

    with open(join(digits_location, 'true_positives_high.pkl'), 'rb') as f:
        true_positives_high = pickle.load(f)

    with open(join(digits_location, 'false_negatives_low.pkl'), 'rb') as f:
        false_negatives_low = pickle.load(f)

    with open(join(digits_location, 'false_negatives_high.pkl'), 'rb') as f:
        false_negatives_high = pickle.load(f)

    return (true_negatives_low,  true_negatives_high),  (false_positives_low,  false_positives_high),  (true_positives_low,  true_positives_high),  (false_negatives_low,  false_negatives_high)

def get_activations(functor,  pairs):
    activations = []
    lengths = []
    for pair in pairs:
        acts = functor([pair[:,  0],  pair[:,  1]])        
        lengths.append(len(acts[1]))
        activations.extend(acts[1])
    return activations, lengths
    
def write_activations_to_file(digits_location,  activations):
    with open(join(digits_location, 'output.tsv'), 'w', newline='') as f_output:
        
        for activation in activations:
            tsv_output = csv.writer(f_output, delimiter='\t')
            tsv_output.writerow(activation)
            
def get_labels(lengths):
    """"
        Calculates labels from activations and stores them in a file. It is necessary to store them in next order:
            true positives, false negatives, true negatives, false positives
    """
    classes = ["tp",  "fn",  "tn",  "fp"]
    classes_detailed = ["tp_low",  "tp_high",  "fn_low",  "fn_high",  "tn_low",  "tn_high",  "fp_low",  "fp_high"]
    labels = []
    labels_detailed = []
    i = 0
    for index, length in enumerate(lengths):

        labels.extend([classes[i]] * length)
        labels_detailed.extend([classes_detailed[index]] * length)
        if index % 2 == 1:
            i += 1

    return (labels,  labels_detailed)

def write_labels_to_file(digits_location,  labels,  labels_detailed):
    with open(join(digits_location, 'metadata.tsv'), 'w') as handle:
        handle.write('{}\t{}\n'.format('Label', 'Label_Detailed'))
        i = 0
        for i in range(len(labels)):
                handle.write('{}\t{}\n'.format(labels[i], labels_detailed[i]))
                i += 1

def get_subsets(pairs, digits):

    subsets_to_return = []
    j = 0
    print('Len: ', len(digits))
    for i in range(len(digits)):
        if i == 2:
            j += 1
        subset_low = create_subset(pairs[j], digits[i][0])
        subset_high = create_subset(pairs[j], digits[i][1])
        subsets_to_return.append(subset_low)
        subsets_to_return.append(subset_high)
        
    return subsets_to_return

def write_output_and_labels(digits_location,  model_location):
    network = load_model(model_location, custom_objects={'contrastive_loss' : contrastive_loss})

    input = [network.layers[0].input,  network.layers[1].input]
    outputs = network.layers[3].input 
    functor = K.function(input, outputs)    
    pairs_tn_fp,  pairs_tp_fn = load_pairs(digits_location)
    (true_negatives_low,  true_negatives_high),  (false_positives_low,  false_positives_high),  (true_positives_low,  true_positives_high),  (false_negatives_low,  false_negatives_high) = load_test_data(digits_location)
    

    pairs = [pairs_tp_fn, pairs_tn_fp]
    digits = [(true_positives_low, true_positives_high), (false_negatives_low, false_negatives_high),
            (true_negatives_low, true_negatives_high), (false_positives_low, false_positives_high)]

    all_pairs = get_subsets(pairs, digits)
    all_activations, lengths = get_activations(functor,  all_pairs)
    write_activations_to_file(digits_location,  all_activations)
    labels,  labels_detailed = get_labels(lengths)
    write_labels_to_file(digits_location,  labels,  labels_detailed)

if __name__ == "__main__":

    for case in test_cases:
        write_output_and_labels(case[0], case[1])
    
