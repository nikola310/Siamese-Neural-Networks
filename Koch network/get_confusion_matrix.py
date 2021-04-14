import pandas as pd

from datetime import datetime
from omniglot_loader import OmniglotLoader
from os import makedirs
from os.path import exists, join
from siamese_network import SiameseNetwork

def write_data_to_file(omniglot, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, csv_file):

        df = pd.DataFrame(columns=['True positives (low)', 'True positives (high)', 'False negatives (low)', 'False negatives (high)', 'True negatives (low)', 'True negatives (high)', 'False positives (low)', 'False positives (high)'])

        for idx in range(len(omg._evaluation_alphabet_list)):
            new_row = pd.DataFrame({'True positives (low)' : [tp_low[idx]],
                                    'True positives (high)' : [tp_high[idx]],
                                    'False negatives (low)' : [fn_low[idx]],
                                    'False negatives (high)' : [fn_high[idx]],
                                    'True negatives (low)' : [tn_low[idx]],
                                    'True negatives (high)' : [tn_high[idx]],
                                    'False positives (low)' : [fp_low[idx]],
                                    'False positives (high)' : [fp_high[idx]]})
            df = pd.concat([df, new_row]).reset_index(drop=True)

        df.loc['Mean'] = df.mean()

        summary_ave_data = df.copy()
        summary_ave_data['Total'] = summary_ave_data.sum(axis=1)

        summary_ave_data.to_csv(csv_file)

if __name__ == "__main__":
    model_w_tf = True
    model_wo_tf = False
    test_w_transformations = True
    test_wo_transformations = True
    eval_classes = 20
    omg = OmniglotLoader()
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    save_dir = './tables/' + run_start_time

    if not exists(save_dir):
        makedirs(save_dir)
    
    if model_wo_tf:
        # model trained without transformations
        sn = SiameseNetwork(model_location='./models/2019-09-24 21-11-23/model.h5')
        
        if test_wo_transformations:
            # Testing without transformations
            omg.use_transformations = False
            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, join(save_dir, 'table_tr_no_tf_te_no_tf.csv'))
        
        if test_w_transformations:
            # Now testing with transformations
            omg.use_transformations = True

            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, join(save_dir, 'table_tr_no_tf_te_tf.csv'))

    if model_w_tf:
        # Model trained with transformations
        sn = SiameseNetwork(model_location='./models/2019-09-25 18-40-12/model.h5')
        
        if test_wo_transformations:
            # Testing without transformations
            omg.use_transformations = False
            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, join(save_dir, 'table_tr_tf_te_no_tf.csv'))
        
        if test_w_transformations:
            # Now testing with transformations
            omg.use_transformations = True

            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, join(save_dir, 'table_tr_tf_te_tf.csv'))
