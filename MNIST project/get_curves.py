import scikitplot as skplt
import sklearn
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists, join
from os import makedirs
import matplotlib.patches as mpatches
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

shape = (28, 28)
num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]
model_wo_tf = './test_models/siamese_model.h5'
model_w_tf = './models/2019-09-28 21-26-51/siamese_model_transformations.h5'

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

if __name__ == '__main__':
    # The data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('Case A')
    # First test case, training without transformations, testing without transformations
    labels = []
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=False)    
    model = load_model(model_wo_tf, custom_objects={'contrastive_loss': contrastive_loss})
    
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    y_pred = np.array(y_pred)
    y_pred = np.flip(y_pred)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(te_y, y_pred)
    auc = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())
    
    print('Case B')
    # Second test case, training without transformations, testing with transformations
    labels = []
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=True)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    y_pred = np.array(y_pred)
    y_pred = np.flip(y_pred)
    fpr_te_tf, tpr_te_tf, _ = sklearn.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc_te_tf = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())

    print('Case C')
    # Third test case, training with transformations, testing with transformations
    model = load_model(model_w_tf, custom_objects={'contrastive_loss': contrastive_loss})
    labels = []
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=True)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    y_pred = np.array(y_pred)
    y_pred = np.flip(y_pred)
    fpr_tf_te_tf, tpr_tf_te_tf, _ = sklearn.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc_tf_te_tf = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())

    print('AUC for \'Training without transformations, testing without transformations\': '+ str(auc))
    print('AUC for \'Training without transformations, testing with transformations\': '+ str(auc_te_tf))
    print('AUC for \'Training with transformations, testing with transformations\': '+ str(auc_tf_te_tf))
    plt.title('ROC Curve')
    patches = []
    patches.append(mpatches.Patch(color='r', label='Case A'))
    patches.append(mpatches.Patch(color='g', label='Case B'))
    patches.append(mpatches.Patch(color='b', label='Case C'))
    plt.legend(handles=patches)
    plt.plot(fpr, tpr, 'r')
    plt.plot(fpr_te_tf, tpr_te_tf, 'g')
    plt.plot(fpr_tf_te_tf, tpr_tf_te_tf, 'b')
    plt.savefig('roc.png')
    plt.show()

    '''
    directory = './projector_data/2019-09-30 22-17-48/Kannada'
    cases = {'model_test' : 'Training without transformations, testing without transformations',
             'model_test_tf' : 'Training without transformations, testing with transformations',
             'model_tf_test_tf' : 'Training with transformations, testing with transformations'}
    for case in cases.keys():
        get_roc_auc(directory, case, cases[case])
    

    
    directories = {'./projector_data/MNIST/projecting_4_no_tf' : 'Training without transformations, testing without transformations', 
                   './projector_data/MNIST/projecting_4_no_tf_8' : 'Training without transformations, testing without transformations (detailed)',
                   './projector_data/MNIST/projecting_4_no_tf_tf' : 'Training without transformations, testing with transformations',
                   './projector_data/MNIST/projecting_4_no_tf_tf_8' : 'Training without transformations, testing with transformations (detailed)', 
                   './projector_data/MNIST/projecting_4_tf' : 'Training with transformations, testing with transformations',
                   './projector_data/MNIST/projecting_4_tf_8' : 'Training with transformations, testing with transformations (detailed)'}

    for directory in directories.keys():
        plot_mnist(directory, directories[directory], 'detailed' in directories[directory])
    '''