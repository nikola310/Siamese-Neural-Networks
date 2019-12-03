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
from train_models import contrastive_loss, create_pairs

shape = (28, 28)
num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]
model_wo_tf = './test_models/siamese_model.h5'
model_w_tf = './test_models/siamese_model_transformations.h5'

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