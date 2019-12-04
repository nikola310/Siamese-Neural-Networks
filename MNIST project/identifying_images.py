import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pickle
from os.path import exists, join
from os import makedirs
from train_models import transform_image, contrastive_loss, compute_accuracy

num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]

model_location = './test_models/'
plot_location_tp_fn = './figures_tp_fn/'
digits_location_tp_fn = './digits_tp_fn/'
plot_location_tn_fp = './figures_tn_fp/'
digits_location_tn_fp = './digits_tn_fp/'

def create_positive_pairs(x, digit_indices, nums=[], test_with_transformations=False):
    pairs = []
    labels = []
    indices = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        indices.append(len(pairs))
        for i in range(0, n, 2):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            if test_with_transformations:
                tr_p1, tr_p2 = transform_image(x[z1], x[z2])
                pairs += [[tr_p1, tr_p2]]
            else:
                pairs += [[x[z1], x[z2]]]
            labels += [1]
    return np.array(pairs), np.array(labels), np.array(indices)

def create_negative_pairs(x, digit_indices, nums=[], test_with_transformations=False):
    pairs = []
    labels = []
    indices = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(10):
        indices.append(len(pairs))
        for i in range(n):

            if test_with_transformations:

                for _ in range(2):
                    
                    inc = random.randrange(1, num_classes)
                    dn = (d + inc) % num_classes
                    nums.append(dn)
                    z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                    
                    tr_p1, tr_p2 = transform_image(x[z1], x[z2])
                    pairs += [[tr_p1, tr_p2]]
            else:
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                nums.append(dn)
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]

                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                nums.append(dn)
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]

            labels += [0, 0]
    return np.array(pairs), np.array(labels), np.array(indices)

def place_in_array(array, dig_idx, index):
    print(dig_idx)
    print(type(dig_idx))
    print(dig_idx.shape)
    print(bruh_moment)
    for j in range(len(dig_idx)):
        if j == 9:
            array[9].append(index)
            break
        elif dig_idx[j] <= index and dig_idx[j+1] > index:
            array[j].append(index)
            break

def print_pairs(digit_list, full_path, pairs):
    j = 0
    while j < len(digit_list) and j <= 5:
        if not exists(full_path):
            makedirs(full_path)
        plt.figure(j)
        plt.subplot(121)
        plt.imshow(pairs[digit_list[j], 0], cmap='gray')
        plt.subplot(122)
        plt.imshow(pairs[digit_list[j], 1], cmap='gray')
        plt.savefig(full_path + str(j) + '.png')
        plt.close(j)
        j += 1

def test_tp_fn(model, plot_location_tp_fn, digits_location_tp_fn, transformations_enabled):
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    labels = []
    te_pairs, te_y, dig_idx = create_positive_pairs(x_test, digit_indices, labels, test_with_transformations=transformations_enabled)
    
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    
    if not exists(plot_location_tp_fn):
        makedirs(plot_location_tp_fn)

    if not exists(digits_location_tp_fn):
        makedirs(digits_location_tp_fn)

    (true_positives_low, true_positives_high), (false_negatives_low, false_negatives_high) = sort_low_and_high_examples_into_arrays(y_pred, dig_idx)

    save_images_of_digits_for_comparison(plot_location_tp_fn, te_pairs, ('True positive (low)', 'False negative (low)'), ('True positive (high)', 'False negative (high)'), 
        true_positives_low, true_positives_high, false_negatives_low, false_negatives_high)

    save_individual_pairs(true_positives_low, true_positives_high, false_negatives_low, false_negatives_high, plot_location_tp_fn,
        ('/true_positives_low/', '/true_positives_high/'), ('/false_negatives_low/', '/false_negatives_high/'), te_pairs)

    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    #serialize pairs
    np.save(digits_location_tp_fn + 'pairs_tp_fn', te_pairs)

    #serialize true positives and false negatives
    with open(digits_location_tp_fn + 'true_positives_low.pkl', 'wb') as f:
        pickle.dump(true_positives_low, f)
    with open(digits_location_tp_fn + 'true_positives_high.pkl', 'wb') as f:
        pickle.dump(true_positives_high, f)
    with open(digits_location_tp_fn + 'false_negatives_low.pkl', 'wb') as f:
        pickle.dump(false_negatives_low, f)
    with open(digits_location_tp_fn + 'false_negatives_high.pkl', 'wb') as f:
        pickle.dump(false_negatives_high, f)

def test_tn_fp(model, plot_location_tn_fp, digits_location_tn_fp, transformations_enabled):
    if not exists(plot_location_tn_fp):
        makedirs(plot_location_tn_fp)

    if not exists(digits_location_tn_fp):
        makedirs(digits_location_tn_fp)

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_negative_pairs, te_y, dig_idx = create_negative_pairs(x_test, digit_indices, test_with_transformations=transformations_enabled)
    y_negative_pred = model.predict([te_negative_pairs[:, 0], te_negative_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_negative_pred)

    (true_negatives_low, true_negatives_high), (false_positives_low, false_positives_high) = sort_low_and_high_examples_into_arrays(y_negative_pred, dig_idx)

    save_images_of_digits_for_comparison(plot_location_tn_fp, te_negative_pairs, ('False positive (low)', 'True negative (low)'), ('False positive (high)', 'True negative (high)'), 
        false_positives_low, false_positives_high, true_negatives_low, true_negatives_high)

    save_individual_pairs(true_negatives_low, true_negatives_high, false_positives_low, false_positives_high, plot_location_tn_fp,
        ('/true_negatives_low/', '/true_negatives_high/'), ('/false_positives_low/', '/false_positives_high/'), te_negative_pairs)

    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    #serialize pairs
    np.save(digits_location_tn_fp + 'pairs_tn_fp', te_negative_pairs)

    #serialize true negatives and false positives
    with open(digits_location_tn_fp + 'true_negatives_low.pkl', 'wb') as f:
        pickle.dump(true_negatives_low, f)
    with open(digits_location_tn_fp + 'true_negatives_high.pkl', 'wb') as f:
        pickle.dump(true_negatives_high, f)
    with open(digits_location_tn_fp + 'false_positives_low.pkl', 'wb') as f:
        pickle.dump(false_positives_low, f)
    with open(digits_location_tn_fp + 'false_positives_high.pkl', 'wb') as f:
        pickle.dump(false_positives_high, f)

def save_images_of_digits_for_comparison(location, pairs, titles_low, titles_high, test_case_0_low, test_case_0_high, test_case_1_low, test_case_1_high):
    """
        Parameters:
            - location - save location
            - pairs - ndarray od pairs
            - titles_low - titles for low examples (i.e. False positive (low))
            - titles_high - titles for high examples (i.e. True negative (high))
    """
    for i in range(10): 
        plt.figure(figsize=(8, 3))
        plt.subplot(141)
        plt.title(titles_low[0])
        plt.axis('off')
        if len(test_case_0_low[i]) > 0:
            plt.imshow(pairs[test_case_0_low[i][0], 1], cmap='gray')
        plt.subplot(142)
        plt.title(titles_high[0])
        plt.axis('off')
        if len(test_case_0_high[i]) > 0:
            plt.imshow(pairs[test_case_0_high[i][0], 1], cmap='gray')
        plt.subplot(143)
        plt.title(titles_low[1])
        plt.axis('off')
        if len(test_case_1_low[i]) > 0:
            plt.imshow(pairs[test_case_1_low[i][0], 1], cmap='gray')
        plt.subplot(144)
        plt.title(titles_high[1])
        if len(test_case_1_high[i]) > 0:
            plt.imshow(pairs[test_case_1_high[i][0], 1], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(location + 'digit_' + str(i) + '.png')

def save_individual_pairs(test_case_0_low, test_case_0_high, test_case_1_low, test_case_1_high, root_location, locations_test_case_0, locations_test_case_1, pairs):
    print('Saving individual pairs...')
    for i in range(10):
        print('Saving ' + str(i) + 's...')
        print_pairs(test_case_0_low[i], root_location +  '/' + str(i) + locations_test_case_0[0], pairs)
        print_pairs(test_case_0_high[i], root_location + '/' + str(i) + locations_test_case_0[1], pairs)
        print_pairs(test_case_1_low[i], root_location + '/' + str(i) + locations_test_case_1[0], pairs)
        print_pairs(test_case_1_high[i], root_location + '/' + str(i) + locations_test_case_1[1], pairs)

def sort_low_and_high_examples_into_arrays(predictions, dig_idx):
    """
        Sorts predictions into arrays, where both of them contain their low and high subclass.
        More spesifically, each prediction is sorted into one of the four classes:
            - below 0.25
            - equal or above 0.25 and below 0.5
            - equal or above 0.5 and below 0.75
            - equal or above 0.75

        Take note that when given true negatives and false positive prediction samples, it returns:
        (true_negatives_low, true_negatives_high), (false_positives_low, false_positives_high),
        but when given true positives and false negatives, return value is:
        (true_positives_low, true_positives_high), (false_negatives_low, false_negatives_high)

        Parameters:
            - predictions: array of predictions
            - dig_idx: digit indices
        Returns:
            - tuples of arrays, organized into low and high subclasses
    """

    class_0_low = [[], [], [], [], [], [], [], [], [], []]
    class_0_high = [[], [], [], [], [], [], [], [], [], []]
    class_1_low = [[], [], [], [], [], [], [], [], [], []]
    class_1_high = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(predictions)):
        if predictions[i] < 0.25:
            place_in_array(class_0_low, dig_idx, i)
        elif predictions[i] >= 0.25 and predictions[i] < 0.5:
            place_in_array(class_0_high, dig_idx, i)
        elif predictions[i] >= 0.5 and predictions[i] < 0.75:
            place_in_array(class_1_low, dig_idx, i)
        elif predictions[i] >= 0.75:
            place_in_array(class_1_high, dig_idx, i)

    return (class_0_low, class_0_high), (class_1_low, class_1_high)

if __name__ == "__main__":

    
    model = load_model(join(model_location, 'siamese_model_transformations.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    test_tp_fn(model, './data/model_w_tf_te/' + plot_location_tp_fn, './data/model_w_tf_te/', False)
    test_tn_fp(model, './data/model_w_tf_te/' + plot_location_tn_fp, './data/model_w_tf_te/', False)
    
    model = load_model(join(model_location, 'siamese_model_transformations.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    test_tp_fn(model, './data/model_w_tf_te_tf/' + plot_location_tp_fn, './data/model_w_tf_te_tf/', True)
    test_tn_fp(model, './data/model_w_tf_te_tf/' + plot_location_tn_fp, './data/model_w_tf_te_tf/', True)
    
    model = load_model(join(model_location, 'siamese_model.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    test_tp_fn(model, './data/model_wo_tf_te/' + plot_location_tp_fn, './data/model_wo_tf_te/', False)
    test_tn_fp(model, './data/model_wo_tf_te/' + plot_location_tn_fp, './data/model_wo_tf_te/', False)

    model = load_model(join(model_location, 'siamese_model.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    test_tp_fn(model, './data/model_wo_tf_te_tf/' + plot_location_tp_fn, './data/model_wo_tf_te_tf/', True)
    test_tn_fp(model, './data/model_wo_tf_te_tf/' + plot_location_tn_fp, './data/model_wo_tf_te_tf/', True)
    