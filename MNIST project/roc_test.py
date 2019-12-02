import scikitplot as skplt
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pickle
from os.path import exists, join
from os import makedirs
from keras.preprocessing.image import ImageDataGenerator

shape = (28, 28)
num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]

model_location = './models/2019-09-28 21-26-51'# './models/2019-09-16 17-53-09/siamese_model_transformations.h5' #'C:/Users/Nikola/Documents/Git/affine_transformations/siamese_model.h5'
plot_location_tp_fn = './figures_tp_fn/'
digits_location_tp_fn = './digits_tp_fn/'
plot_location_tn_fp = './figures_tn_fp/'
digits_location_tn_fp = './digits_tn_fp/'

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def transform_image(image_1, image_2, img_transformer):

    transformations = {}
    # Calculating transformation
    if np.random.uniform(low=0, high=1) < 0.5:
        theta = np.random.uniform(low=rotation_range[0], high=rotation_range[1])
        transformations['theta'] = theta
    if np.random.uniform(low=0, high=1) < 0.5:
        tx = np.random.uniform(low=shift_range[0], high=shift_range[1])
        transformations['tx'] = tx
    if np.random.uniform(low=0, high=1) < 0.5:
        ty = np.random.uniform(low=shift_range[0], high=shift_range[1])
        transformations['ty'] = ty
    if np.random.uniform(low=0, high=1) < 0.5:
        zx = np.random.uniform(low=scale_range[0], high=scale_range[1])
        transformations['zx'] = zx
    if np.random.uniform(low=0, high=1) < 0.5:
        zy = np.random.uniform(low=scale_range[0], high=scale_range[1])
        transformations['zy'] = zy
    if np.random.uniform(low=0, high=1) < 0.5:
        shear = np.random.uniform(low=shear_range[0], high=shear_range[1])
        transformations['shear'] = shear
    
    image_1 = np.expand_dims(image_1, axis=-1)
    image_2 = np.expand_dims(image_2, axis=-1)
    image_1 = img_transformer.apply_transform(image_1, transformations)
    image_2 = img_transformer.apply_transform(image_2, transformations) 
    return image_1[:,:,0], image_2[:,:,0] 

def create_pairs(x, digit_indices, nums=[], transform=False):
    '''
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
    '''
    pairs = [] #np.array([])
    labels = []
    gen = ImageDataGenerator()
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            if transform:
                # performing transformation
                # positive pairs transformation
                tr_p1, tr_p2 = transform_image(x[z1], x[z2], gen)

            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            nums.append(dn)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
            
            if transform:
                # performing transformation
                # negative pairs transformation
                tr_n1, tr_n2 = transform_image(x[z1], x[z2], gen)
                pairs += [[tr_p1, tr_p2]]
                pairs += [[tr_n1, tr_n2]]
                labels += [1, 0]
    if transform:
        return np.array(pairs, ndmin=3), np.array(labels)
    else:
        return np.array(pairs), np.array(labels)

def create_positive_pairs(x, digit_indices, nums=[], test_with_transformations=False):
    pairs = []
    labels = []
    indices = []
    transformer = ImageDataGenerator()
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        indices.append(len(pairs))
        for i in range(0, n, 2):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            if test_with_transformations:
               
                tr_p1, tr_p2 = transform_image(x[z1], x[z2], transformer)
                pairs += [[tr_p1, tr_p2]]
            else:
                pairs += [[x[z1], x[z2]]]
            labels += [1]
    return np.array(pairs), np.array(labels), np.array(indices)

def create_negative_pairs(x, digit_indices, nums=[], test_with_transformations=False):
    pairs = []
    labels = []
    indices = []
    transformer = ImageDataGenerator()
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
                    tr_p1, tr_p2 = transform_image(x[z1], x[z2], transformer)
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
    for j in range(len(dig_idx)):
        if j == 9:
            array[9].append(index)
            break
        elif dig_idx[j] <= index and dig_idx[j+1] > index:
            array[j].append(index)
            break
    
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def print_pairs(digit_list, full_path, te_pairs):
    for j in range(len(digit_list)):
        if not exists(full_path):
            makedirs(full_path)
        plt.figure(j)
        plt.subplot(121)
        plt.imshow(te_pairs[digit_list[j], 0], cmap='gray')
        plt.subplot(122)
        plt.imshow(te_pairs[digit_list[j], 1], cmap='gray')
        plt.savefig(full_path + str(j) + '.png')
        plt.close(j)
        if j == 10:
            break

def test_tp_fn(model, plot_location_tp_fn, digits_location_tp_fn, transformations_enabled):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

    true_positives_low = [[], [], [], [], [], [], [], [], [], []]
    true_positives_high = [[], [], [], [], [], [], [], [], [], []]
    false_negatives_low = [[], [], [], [], [], [], [], [], [], []]
    false_negatives_high = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(y_pred)):
        if y_pred[i] < 0.25:
            place_in_array(true_positives_low, dig_idx, i)
        elif y_pred[i] >= 0.25 and y_pred[i] < 0.5:
            place_in_array(true_positives_high, dig_idx, i)
        elif y_pred[i] >= 0.5 and y_pred[i] < 0.75:
            place_in_array(false_negatives_low, dig_idx, i)
        elif y_pred[i] >= 0.75:
            place_in_array(false_negatives_high, dig_idx, i)

    for i in range(10):
        plt.figure(0, figsize=(8, 3))
        plt.subplot(141)
        plt.title('True positive (low)')
        plt.axis('off')
        plt.imshow(te_pairs[true_positives_low[i][0], 1], cmap='gray')
        plt.subplot(142)
        plt.title('True positive (high)')
        plt.axis('off')
        plt.imshow(te_pairs[true_positives_high[i][0], 1], cmap='gray')
        plt.subplot(143)
        plt.title('False negative (low)')
        plt.axis('off')
        if len(false_negatives_low[i]) > 0:
            plt.imshow(te_pairs[false_negatives_low[i][0], 1], cmap='gray')
        plt.subplot(144)
        plt.title('False negative (high)')
        plt.imshow(te_pairs[false_negatives_high[i][0], 1], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_location_tp_fn + 'digit_' + str(i) + '.png')
        plt.close(0)

        print('Saving ' + str(i) + 's...')
        print_pairs(true_positives_low[i], plot_location_tp_fn + '/' + str(i) + '/true_positives_low/', te_pairs)
        print_pairs(true_positives_high[i], plot_location_tp_fn + '/' + str(i) + '/true_positives_high/', te_pairs)
        print_pairs(false_negatives_low[i], plot_location_tp_fn + '/' + str(i) + '/false_negatives_low/', te_pairs)
        print_pairs(false_negatives_high[i], plot_location_tp_fn + '/' + str(i) + '/false_negatives_high/', te_pairs)

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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_negative_pairs, te_y, dig_idx = create_negative_pairs(x_test, digit_indices, test_with_transformations=transformations_enabled)
    y_negative_pred = model.predict([te_negative_pairs[:, 0], te_negative_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_negative_pred)

    true_negatives_low = [[], [], [], [], [], [], [], [], [], []]
    true_negatives_high = [[], [], [], [], [], [], [], [], [], []]
    false_positives_low = [[], [], [], [], [], [], [], [], [], []]
    false_positives_high = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(y_negative_pred)):
        if y_negative_pred[i] < 0.25:
            place_in_array(false_positives_low, dig_idx, i)
        elif y_negative_pred[i] >= 0.25 and y_negative_pred[i] < 0.5:
            place_in_array(false_positives_high, dig_idx, i)
        elif y_negative_pred[i] >= 0.5 and y_negative_pred[i] < 0.75:
            place_in_array(true_negatives_low, dig_idx, i)
        elif y_negative_pred[i] >= 0.75:
            place_in_array(true_negatives_high, dig_idx, i)

    for i in range(10):
        plt.figure(figsize=(8, 3))
        plt.subplot(141)
        plt.title('False positive (low)')
        plt.axis('off')
        if len(false_positives_low[i]) > 0:
            plt.imshow(te_negative_pairs[false_positives_low[i][0], 1], cmap='gray')
        plt.subplot(142)
        plt.title('False positive (high)')
        plt.axis('off')
        if len(false_positives_high[i]) > 0:
            plt.imshow(te_negative_pairs[false_positives_high[i][0], 1], cmap='gray')
        plt.subplot(143)
        plt.title('True negative (low)')
        plt.axis('off')
        if len(true_negatives_low[i]) > 0:
            plt.imshow(te_negative_pairs[true_negatives_low[i][0], 1], cmap='gray')
        plt.subplot(144)
        plt.title('True negative (high)')
        if len(true_negatives_high[i]) > 0:
            plt.imshow(te_negative_pairs[true_negatives_high[i][0], 1], cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(plot_location_tn_fp + 'digit_' + str(i) + '.png')
    
    print('Saving individual pairs...')
    for i in range(10):
        print('Saving ' + str(i) + 's...')
        print_pairs(true_negatives_low[i], plot_location_tn_fp + '/' + str(i) + '/true_negatives_low/', te_negative_pairs)
        print_pairs(true_negatives_high[i], plot_location_tn_fp + '/' + str(i) + '/true_negatives_high/', te_negative_pairs)
        print_pairs(false_positives_low[i], plot_location_tn_fp + '/' + str(i) + '/false_positives_low/', te_negative_pairs)
        print_pairs(false_positives_high[i], plot_location_tn_fp + '/' + str(i) + '/false_positives_high/', te_negative_pairs)

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

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]

    model = load_model(join(model_location, 'siamese_model_transformations.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    labels = []
    transformations=True
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=False)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    fpr, tpr, thresholds = skplt.metrics.roc_curve(te_y, y_pred, pos_label=0)
    plt.plot(fpr,tpr)
    plt.show()