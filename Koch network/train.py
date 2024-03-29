import argparse
import tensorflow as tf

from omniglot_loader import OmniglotLoader
from siamese_network import SiameseNetwork

'''
    Script for training model with transformations
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Train network',
                                    formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_transformations', 
                        action = 'store_true', 
                        default = False,
                        help = 'train model with affine transformations')
    parser.add_argument('--gpu', 
                        default = 0, type = int,
                        help = 'specify which GPU to use')
    parser.add_argument('--epochs',
                        default = 10, type = int,
                        help = 'specify number of epochs')
    parser.add_argument('--memory_limit',
                        default = 7168, type = int,
                        help = 'specify memory limit for GPU')

    args = parser.parse_args()
    return args

def run_program(use_transformations=False, epochs=10):
    omg = OmniglotLoader(use_transformations=use_transformations)

    sn = SiameseNetwork()

    sn.train(omg, epochs)

def main():
    args = parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[args.gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            with tf.device('GPU:' + str(args.gpu)):
                run_program(args.use_transformations, args.epochs)
        except RuntimeError as e:
            print(e)
    else:
        run_program(args.use_transformations, args.epochs)

if __name__ == "__main__":
    main()
