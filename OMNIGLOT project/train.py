import tensorflow as tf
from omniglot_loader import OmniglotLoader
from siamese_network import SiameseNetwork

'''
    Script for training model without transformations
'''
if __name__ == "__main__":

    #gpus = tf.config.experimental.list_physical_devices('GPU')
    
    omg = OmniglotLoader(use_transformations=False)

    sn = SiameseNetwork()

    sn.train(omg, 200)
    '''
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            with tf.device('GPU:0'):
                omg = OmniglotLoader(use_transformations=True)

                sn = SiameseNetwork()

                sn.train(omg, 200)
        except RuntimeError as e:
            print(e)
    else:
        sn.train(omg, 2)
    '''
