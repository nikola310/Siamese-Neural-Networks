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
    #plt.gca().set_aspect('equal', 'datalim')
    patches = []
    for idx, col in enumerate(colors_detailed):
        patches.append(mpatches.Patch(color=col, label=classes_dict[uniques_detailed[idx]]))

    plt.legend(handles=patches)
    plt.title(title, fontsize=30);
    plt.savefig(join(directory, 'plot_detailed.png'))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    plot('./data/model_w_tf_te', 'Training with transformations, testing without transformations')
    plot('./data/model_w_tf_te_tf', 'Training with transformations, testing with transformations')
    plot('./data/model_wo_tf_te', 'Training without transformations, testing without transformations')
    plot('./data/model_wo_tf_te_tf', 'Training without transformations, testing with transformations')