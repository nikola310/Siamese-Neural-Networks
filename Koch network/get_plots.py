import pandas as pd
import umap
from os.path import join
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

classes_dict = {'tp':'True positives', 'fn':'False negatives', 'tn':'True negatives', 'fp':'False positives',
                'tp_low':'True positives (low)', 'fn_low':'False negatives (low)', 'tn_low':'True negatives (low)', 'fp_low':'False positives (low)',
                'tp_high':'True positives (high)', 'fn_high':'False negatives (high)', 'tn_high':'True negatives (high)', 'fp_high':'False positives (high)'}
colors = ['red','green','blue','purple']
colors_detailed = ['red','green','blue','purple', 'pink', 'orange', 'brown', 'turquoise']

def plot(directory, case, title):
    meta = 'metadata_' + case + '.tsv'
    out = 'output_' + case + '.tsv'
    
    x = pd.read_csv(join(directory, out), sep='\t', header=None)
    labels = pd.read_csv(join(directory, meta), sep='\t')
    labels.Label, uniques = pd.factorize(labels.Label)
    labels.Label_Detailed, uniques_detailed =  pd.factorize(labels.Label_Detailed)
    print(uniques_detailed)
    reducer = umap.UMAP(verbose=True)
    embedding = reducer.fit_transform(x)
    
    plt.figure(figsize=(15,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.Label, cmap=matplotlib.colors.ListedColormap(colors))

    patches = []
    for idx, col in enumerate(colors):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=24)
    plt.savefig(join(directory, 'plot_' + case + '.png'))
    
    title += ' (detailed)'
    
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(15,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.Label_Detailed, cmap=matplotlib.colors.ListedColormap(colors_detailed))
    #plt.gca().set_aspect('equal', 'datalim')
    patches = []
    for idx, col in enumerate(colors_detailed):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques_detailed[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=24)
    plt.savefig(join(directory, 'plot_' + case + '_detailed.png'))
    plt.clf()
    plt.close()

def plot_mnist(directory, title, detailed):
    meta = 'metadata.tsv'
    out = 'output.tsv'
    if detailed:
        colors_mnist = colors_detailed
    else:
        colors_mnist = colors
    
    x = pd.read_csv(join(directory, out), sep='\t', header=None)
    labels = pd.read_csv(join(directory, meta), sep='\t')
    labels.Label, uniques = pd.factorize(labels.Label)
    
    reducer = umap.UMAP(verbose=True)
    embedding = reducer.fit_transform(x)
    
    plt.figure(figsize=(15, 7))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.Label, cmap=matplotlib.colors.ListedColormap(colors_mnist))

    patches = []
    for idx, col in enumerate(colors_mnist):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=24)
    plt.savefig(join(directory, 'plot.png'))
    plt.clf()
    plt.close()

if __name__ == '__main__':
    
    directory = './projector_data/2019-09-30 22-17-48/Kannada'
    cases = {'model_test' : 'Training without transformations, testing without transformations',
             'model_test_tf' : 'Training without transformations, testing with transformations',
             'model_tf_test_tf' : 'Training with transformations, testing with transformations'}
    for case in cases.keys():
        plot(directory, case, cases[case])