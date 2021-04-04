# -*- coding: utf-8 -*-
'''
    Source: https://keras.io/examples/mnist_siamese/
'''

import numpy as np
import tensorflow as tf
import random
from os import makedirs
from os.path import exists, join
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout,  Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

num_classes = 10
epochs = 12
batch_size = 128
number_of_splits = 5

input_shape = (28, 28)
number_of_epochs = 12

rotation_range = [-12, 12]
shear_range = [-12, 12]
scale_range = [0.9, 1.4]
shift_range = [-4, 4]

image_transformer = ImageDataGenerator()

kmnist_test_images_file = 'kmnist-test-imgs.npz'
kmnist_test_labels_file = 'kmnist-test-labels.npz'
kmnist_train_images_file = 'kmnist-train-imgs.npz'
kmnist_train_labels_file = 'kmnist-train-labels.npz'

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
            tf.compat.v1.summary.histogram('histogram', var)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def transform_image(image_1, image_2):
    transformations = {}
    # Calculating transformation
    if np.random.uniform(low=0, high=1) < 0.5:
        theta = np.random.uniform(low=rotation_range[0], high=rotation_range[1])
        transformations['theta'] = theta
    if np.random.uniform(low=0, high=1) < 0.5:
        tx = np.random.uniform(low=shift_range[0], high=shift_range[1])
        transformations['tx'] = tx
    if np.random.uniform(low=0, high=1) < 0.5:
        ty = np.random.uniform(low=shift_range[0], high=shift_range[1])
        transformations['ty'] = ty
    if np.random.uniform(low=0, high=1) < 0.5:
        zx = np.random.uniform(low=scale_range[0], high=scale_range[1])
        transformations['zx'] = zx
    if np.random.uniform(low=0, high=1) < 0.5:
        zy = np.random.uniform(low=scale_range[0], high=scale_range[1])
        transformations['zy'] = zy
    if np.random.uniform(low=0, high=1) < 0.5:
        shear = np.random.uniform(low=shear_range[0], high=shear_range[1])
        transformations['shear'] = shear
    
    image_1 = np.expand_dims(image_1, axis=-1)
    image_2 = np.expand_dims(image_2, axis=-1)
    image_1 = image_transformer.apply_transform(image_1, transformations)
    image_2 = image_transformer.apply_transform(image_2, transformations) 
    return image_1[:,:,0], image_2[:,:,0] 

def create_pairs(x, digit_indices, nums=[], transform=False):
    '''
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []

    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]

            if transform:
                # positive pairs transformation
                tr_p1, tr_p2 = transform_image(x[z1], x[z2])

            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            nums.append(dn)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1.0, 0.0]
            
            if transform:
                # negative pairs transformation
                tr_n1, tr_n2 = transform_image(x[z1], x[z2])

                pairs += [[tr_p1, tr_p2]]
                pairs += [[tr_n1, tr_n2]]
                labels += [1.0, 0.0]
    if transform:
        return np.array(pairs, ndmin=3), np.array(labels)
    else:
        return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu', name="features")(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def get_data(training_data=True):

    if training_data:
        x = np.load(kmnist_train_images_file)['arr_0']
        y = np.load(kmnist_train_labels_file)['arr_0']
    else:
        x = np.load(kmnist_test_images_file)['arr_0']
        y = np.load(kmnist_test_labels_file)['arr_0']

    x = x.astype('float32')
    x /= 255

    return (x, y)

def prepare_data_for_work(transformations):
    (x_train, y_train) = get_data()

    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    labels = []
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, transform=transformations)
    (x_test, y_test) = get_data(False)
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=transformations)

    return (tr_pairs, tr_y), (te_pairs, te_y)

def prepare_data_for_training(transformations):
    (x_train, y_train) = get_data()

    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    labels = []
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, transform=transformations)

    return (tr_pairs, tr_y)

def prepare_data_for_testing(transformations):
    (x_test, y_test) = get_data(False)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    labels = []
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=transformations)
    
    return (te_pairs, te_y)


def compute_final_accuracy(model, tr_pairs, tr_y, te_pairs, te_y):

    if tr_pairs is not None and tr_y is not None: 
        y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]], batch_size=batch_size)

        tr_acc = compute_accuracy(tr_y, y_pred)
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    
    if te_pairs is not None and te_y is not None:
        y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]], batch_size=batch_size)
        te_acc = compute_accuracy(te_y, y_pred)

        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def create_model():
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape, name='lambda')([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    model.compile(loss=contrastive_loss, optimizer='sgd', metrics=['accuracy'])

    return model

def get_tensorboards(model, model_dir, transformations):
    
    if transformations:
        logs = model_dir + "/logs-tr/"
    else:
        logs = model_dir + "/logs/"
    tensorboard = TensorBoard(
        log_dir=logs,
        histogram_freq=0,
        batch_size=20,
        write_graph=True,
        write_grads=True)
    tensorboard.set_model(model)
    '''
    '''
    return tensorboard

def train_model(model_dir, transformations=False):
    
    (tr_pairs, tr_y), (te_pairs, te_y) = prepare_data_for_work(transformations)

    model = create_model()
    tensorboard = get_tensorboards(model, model_dir, transformations)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size,
            epochs=epochs, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), callbacks=[tensorboard])

    compute_final_accuracy(model, tr_pairs, tr_y, te_pairs, te_y)

    return model

def train_model2(model_dir, transformations=False):
    
    (tr_pairs, tr_y) = prepare_data_for_training(transformations)
    print('Create model')
    model = create_model()
    tensorboard = get_tensorboards(model, model_dir, transformations)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard])

    #compute_final_accuracy(model, tr_pairs, tr_y, None, None)
    #y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]], batch_size=batch_size)

    #tr_acc = compute_accuracy(tr_y, y_pred)
    #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    #(te_pairs, te_y) = prepare_data_for_testing(transformations)

    #compute_final_accuracy(model, None, None, te_pairs, te_y)

    return model

if __name__ == "__main__":
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    model_dir = 'models/' + run_start_time

    if not exists(model_dir):
        makedirs(model_dir)
    
    # Training model without transformations
    #model = train_model(model_dir)
    #model.save(join(model_dir, 'siamese_model.h5'))
    
    # Training model with transformations
    model_2 = train_model2(model_dir, True)
    model_2.save(join(model_dir, 'siamese_model_transformations.h5'))
