'''
    Source: https://keras.io/examples/mnist_siamese/
'''

import numpy as np
import tensorflow as tf
import random
from os import makedirs
from os.path import exists, join
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 10
epochs = 12
batch_size = 128

# input image dimensions
input_shape = (28, 28)

rotation_range = [-12, 12]
shear_range = [-12, 12]
scale_range = [0.9, 1.4]
shift_range = [-4, 4]

image_transformer = ImageDataGenerator()

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
    x = Dense(128, activation='relu', name='dense_proba')(x)    
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


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def get_training_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)

def prepare_data_for_training(transformations):

    (x_train, y_train), (x_test, y_test) = get_training_data()

    # Create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    labels = []
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, transform=transformations)
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=transformations)

    return (tr_pairs, tr_y), (te_pairs, te_y)

def compute_final_accuracy(model, tr_pairs, tr_y, te_pairs, te_y):

    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])

    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


def train_model(model_dir, transformations=False):
    
    (tr_pairs, tr_y), (te_pairs, te_y) = prepare_data_for_training(transformations)

    # Network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape, name='lambda')([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    # Train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    compute_final_accuracy(model, tr_pairs, tr_y, te_pairs, te_y)

    return model


if __name__ == "__main__":
    run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
    model_dir = 'models/' + run_start_time

    if not exists(model_dir):
        makedirs(model_dir)
    
    model = train_model(model_dir)
    model.save(join(model_dir, 'siamese_model.h5'))
    
    # Training model with transformation
    model = train_model(model_dir, True)
    model.save(join(model_dir, 'siamese_model_transformations.h5'))