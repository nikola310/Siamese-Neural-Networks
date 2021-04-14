import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import sklearn

from omniglot_loader import OmniglotLoader
from siamese_network import SiameseNetwork

model_wo_tf = './trained_models/wo_transform/model.h5'
model_w_tf = './trained_models/w_transform/model.h5'

if __name__ == '__main__':

    # First test case, training without transformations, testing without transformations
    omg = OmniglotLoader(use_transformations=False)
    network = SiameseNetwork(model_location=model_wo_tf)
    
    y_pred, te_y = network.get_predictions(omg)
    y_pred = np.array(y_pred)
    te_y = np.array(te_y)
    fpr, tpr, _ = skplt.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())
    
    # Second test case, training without transformations, testing with transformations
    omg = OmniglotLoader(use_transformations=True)
    y_pred, te_y = network.get_predictions(omg)
    y_pred = np.array(y_pred)
    te_y = np.array(te_y)
    fpr_te_tf, tpr_te_tf, _ = skplt.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc_te_tf = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())

    # Third test case, training with transformations, testing with transformations
    omg = OmniglotLoader(use_transformations=True)
    network = SiameseNetwork(model_location=model_w_tf)
    y_pred, te_y = network.get_predictions(omg)
    y_pred = np.array(y_pred)
    te_y = np.array(te_y)
    fpr_tf_te_tf, tpr_tf_te_tf, _ = skplt.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc_tf_te_tf = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())

    print('AUC for case \'Training without transformations, testing without transformations\': '+ str(auc))
    print('AUC for case \'Training without transformations, testing with transformations\': '+ str(auc_te_tf))
    print('AUC for case \'Training with transformations, testing with transformations\': '+ str(auc_tf_te_tf))
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
