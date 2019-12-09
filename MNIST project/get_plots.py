import pandas as pd
import umap
import sklearn
from os.path import join
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from train_models import prepare_data_for_work, contrastive_loss
from tensorflow.keras.models import load_model
import numpy as np

classes_dict = {'tp':'True positives', 'fn':'False negatives', 'tn':'True negatives', 'fp':'False positives',
                'tp_low':'True positives (low)', 'fn_low':'False negatives (low)', 'tn_low':'True negatives (low)', 'fp_low':'False positives (low)',
                'tp_high':'True positives (high)', 'fn_high':'False negatives (high)', 'tn_high':'True negatives (high)', 'fp_high':'False positives (high)'}
colors = ['red','green','blue','purple']
colors_detailed = ['red','green','blue','purple', 'pink', 'orange', 'brown', 'turquoise']

shape = (28, 28)
num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]
model_wo_tf = './test_models/siamese_model.h5'
model_w_tf = './test_models/siamese_model_transformations.h5'

def plot(directory, title):
    meta = 'metadata.tsv'
    out = 'output.tsv'
    
    x = pd.read_csv(join(directory, out), sep='\t', header=None)
    labels = pd.read_csv(join(directory, meta), sep='\t')
    labels.Label, uniques = pd.factorize(labels.Label)
    labels.Label_Detailed, uniques_detailed=  pd.factorize(labels.Label_Detailed)
    
    reducer = umap.UMAP(verbose=True)
    embedding = reducer.fit_transform(x)
    
    plt.figure(figsize=(20,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.Label, cmap=matplotlib.colors.ListedColormap(colors))

    patches = []
    for idx, col in enumerate(colors):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=30);
    plt.savefig(join(directory, 'plot.png'))
    
    title += ' (detailed)'
    
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(20,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.Label_Detailed, cmap=matplotlib.colors.ListedColormap(colors_detailed))

    patches = []
    for idx, col in enumerate(colors_detailed):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques_detailed[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=30);
    plt.savefig(join(directory, 'plot_detailed.png'))
    plt.clf()
    plt.close()

def get_roc_and_auc_curves():


    print('Case A')
    # First test case, training without transformations, testing without transformations
    (_, _), (te_pairs, te_y) = prepare_data_for_work(False)
    model = load_model(model_wo_tf, custom_objects={'contrastive_loss': contrastive_loss})
    
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    y_pred = np.array(y_pred)
    y_pred = np.flip(y_pred)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(te_y, y_pred)
    auc = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())
    
    print('Case B')
    # Second test case, training without transformations, testing with transformations
    (_, _), (te_pairs, te_y) = prepare_data_for_work(True)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    y_pred = np.array(y_pred)
    y_pred = np.flip(y_pred)
    fpr_te_tf, tpr_te_tf, _ = sklearn.metrics.roc_curve(te_y.flatten(), y_pred.flatten())
    auc_te_tf = sklearn.metrics.roc_auc_score(te_y.flatten(), y_pred.flatten())

    print('Case C')
    # Third test case, training with transformations, testing with transformations
    model = load_model(model_w_tf, custom_objects={'contrastive_loss': contrastive_loss})
    (_, _), (te_pairs, te_y) = prepare_data_for_work(True)
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


if __name__ == '__main__':
    #get_roc_and_auc_curves()

    plot('./data/model_w_tf_te', 'Training with transformations, testing without transformations')
    plot('./data/model_w_tf_te_tf', 'Training with transformations, testing with transformations')
    plot('./data/model_wo_tf_te', 'Training without transformations, testing without transformations')
    plot('./data/model_wo_tf_te_tf', 'Training without transformations, testing with transformations')
