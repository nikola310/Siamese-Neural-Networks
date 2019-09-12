import tensorflow as tf
from tensorflow.keras.models import load_model
from omniglot_loader import OmniglotLoader
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists, join
from siamese_network import SiameseNetwork
import pandas as pd

def write_data_to_file(omniglot, class_num, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, csv_file):
        '''
            Function for writing data to csv file.

            Arguments:
                - omniglot = instance of OmniglotLoader
                - class_num = number of classes
                - tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high = array with numbers of true positive, false negative, etc. classes
                - csv_file = output file
        '''
        df = pd.DataFrame(columns=['True positives (low)', 'True positives (high)', 'False negatives (low)', 'False negatives (high)', 'True negatives (low)', 'True negatives (high)', 'False positives (low)', 'False positives (high)'])

        for idx in range(class_num):
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
    model_w_tf = False
    model_wo_tf = True
    test_w_transformations = False
    test_wo_transformations = True
    eval_classes = 20
    omg = OmniglotLoader()

    '''
    imgs, labels = omg.get_negative_batch(False) #omg.get_positive_batch(False)
    
    plt.figure(0, figsize=(8, 3))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(imgs[2, 0].reshape(105, 105), cmap='binary')
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(imgs[2, 1].reshape(105, 105), cmap='binary')
    plt.show()
    print(omg.get_current_symbol_idx())
    
    imgs, labels = omg.get_negative_batch(False)
    print(omg.get_current_symbol_idx())
    plt.figure(0, figsize=(8, 3))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(imgs[2, 0].reshape(105, 105), cmap='binary')
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(imgs[2, 1].reshape(105, 105), cmap='binary')
    plt.show()
    '''
    if model_wo_tf:
        # Model trained without transformations
        sn = SiameseNetwork(model_location='C:/Users/Nikola/Documents/Git/Siamese-Neural-Networks/models/2019-09-09 18-00-26/model.h5')
        
        if test_wo_transformations:
            # Testing without transformations
            omg.set_use_transformations(False)
            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, eval_classes, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, 'table_tr_no_tf_te_no_tf.csv')
        
        if test_w_transformations:
            # Now testing with transformations
            omg.set_use_transformations(True)

            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, eval_classes, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, 'table_tr_no_tf_te_tf.csv')

    if model_w_tf:
        # Model trained with transformations
        sn = SiameseNetwork(model_location='')
        
        if test_wo_transformations:
            # Testing without transformations
            omg.set_use_transformations(False)
            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, eval_classes, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, 'table_tr_tf_te_no_tf.csv')
        
        if test_w_transformations:
            # Now testing with transformations
            omg.set_use_transformations(True)

            tp_low, tp_high, fn_low, fn_high = sn.test_tp_fn(omg)
            tn_low, tn_high, fp_low, fp_high = sn.test_tn_fp(omg)
            write_data_to_file(omg, eval_classes, tp_low, tp_high, fn_low, fn_high, tn_low, tn_high, fp_low, fp_high, 'table_tr_tf_te_tf.csv')