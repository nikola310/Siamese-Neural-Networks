from os.path import join
import pickle
import pandas as pd

folders = ['./data/model_w_tf_te', './data/model_w_tf_te_tf', './data/model_wo_tf_te', './data/model_wo_tf_te_tf']

if __name__ == "__main__":
    
    for f in folders:

        with open(join(f, 'true_negatives_low.pkl'), 'rb') as data:
            true_negatives_low = pickle.load(data)

        with open(join(f, 'true_negatives_high.pkl'), 'rb') as data:
            true_negatives_high = pickle.load(data)

        with open(join(f, 'false_positives_low.pkl'), 'rb') as data:
            false_positives_low = pickle.load(data)

        with open(join(f, 'false_positives_high.pkl'), 'rb') as data:
            false_positives_high = pickle.load(data)

        with open(join(f, 'true_positives_low.pkl'), 'rb') as data:
            true_positives_low = pickle.load(data)

        with open(join(f, 'true_positives_high.pkl'), 'rb') as data:
            true_positives_high = pickle.load(data)

        with open(join(f, 'false_negatives_low.pkl'), 'rb') as data:
            false_negatives_low = pickle.load(data)

        with open(join(f, 'false_negatives_high.pkl'), 'rb') as data:
            false_negatives_high = pickle.load(data)

        df = pd.DataFrame(columns=['True positives (low)', 'True positives (high)', 'False negatives (low)', 'False negatives (high)', 'True negatives (low)', 'True negatives (high)', 'False positives (low)', 'False positives (high)'])

        for i in range(10):
            new_row = pd.DataFrame({'True positives (low)' : [len(true_positives_low[i])],
                                    'True positives (high)' : [len(true_positives_high[i])],
                                    'False negatives (low)' : [len(false_negatives_low[i])],
                                    'False negatives (high)' : [len(false_negatives_high[i])],
                                    'True negatives (low)' : [len(true_negatives_low[i])],
                                    'True negatives (high)' : [len(true_negatives_high[i])],
                                    'False positives (low)' : [len(false_positives_low[i])],
                                    'False positives (high)' : [len(false_positives_high[i])]})
            df = pd.concat([df, new_row]).reset_index(drop=True)

        df.loc['Mean'] = df.mean()

        summary_ave_data = df.copy()
        summary_ave_data['Total'] = summary_ave_data.sum(axis=1)

        summary_ave_data.to_csv(join(f, 'table.csv'))
        
    print('Done.')