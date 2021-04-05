import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
import os
from datetime import datetime
import numpy as np

class SiameseNetwork:

    def __init__(self, input_shape=(105, 105, 1), batch=20, model_location=None):
        '''
            Constructor for the SiameseNetworks class.

            Arguments:
                - input_shape = shape of input images
                - batch = size of batches
                - model_location = location of trained model to be loaded
        '''
        self.input_shape = input_shape
        self.l2_penalization = 1e-2
        if model_location is not None:
            self.model = load_model(model_location)
        else:
            self.model = self.__get_siamese_model()
        self.batch = batch
        run_start_time = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        self.__save_dir = './models/' + run_start_time
        self.__train_dir = './logs/' + run_start_time + '/training'
        self.__test_dir = './logs/' + run_start_time + '/test'
        self.__confusion_matrix_dir = './confusion_matrix_data/' + run_start_time

    ####################################################################
    # Getters and setters                                              #
    ####################################################################
    def get_current_epoch(self):
        return self.__current_epoch

    def set_current_epoch(self, epoch):
        self.__current_epoch = epoch

    def __get_siamese_model(self):
        '''
            Creates siamese neural network

            Returns:
                - model = instance of siamese neural network
        '''

        # First create base convolutional network
        base_network = Sequential()
        base_network.add(Conv2D(filters=64, kernel_size=(10, 10), activation='relu', 
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            input_shape=self.input_shape, kernel_regularizer=l2(self.l2_penalization),  name='Conv_layer_1'))
        base_network.add(MaxPool2D())
        base_network.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu',
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), 
            kernel_regularizer=l2(self.l2_penalization), name='Conv_layer_2'))
        base_network.add(MaxPool2D())
        base_network.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu',
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), 
            kernel_regularizer=l2(self.l2_penalization), name='Conv_layer_3'))
        base_network.add(MaxPool2D())
        base_network.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu',
            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2), 
            kernel_regularizer=l2(self.l2_penalization), name='Conv_layer_4'))
        base_network.add(Flatten())
        base_network.add(Dense(units=4096, activation='sigmoid', 
            kernel_initializer=RandomNormal(mean=0.0, stddev=2e-1),
            bias_initializer=RandomNormal(mean=0.5, stddev=1e-2),
            kernel_regularizer=l2(self.l2_penalization), name='Dense_layer_1'))

        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        encoded_image_left = base_network(left_input)
        encoded_image_right = base_network(right_input)
        
        # Defining L1 distance layer
        distance_layer = Lambda(lambda encoding: K.abs(encoding[0] - encoding[1]))
        l1_distance = distance_layer([encoded_image_left, encoded_image_right])
        
        # Prediction layer
        prediction = Dense(1, activation='sigmoid')(l1_distance)
        model = tf.keras.Model([left_input, right_input], [prediction])
        model.compile(loss='binary_crossentropy', metrics=['binary_accuracy', 'mse', 'mae', 'acc'], optimizer='sgd')
        return model

    ####################################################################
    # Train and testing methods                                        #
    ####################################################################
    def train(self, omniglot, epoch_num):
        '''
            Trains the network for given number of epochs

            Arguments:
                - omniglot = instance of OmniglotLoader
                - epoch_num = number of epochs
        '''
        epoch_id = 0

        self._create_training_directories()

        tensorboard, tensorboard_eval = self._get_tensorboards()

        omniglot._training = True

        print('Training started.')
        while True:
            if epoch_id >= epoch_num:
                break

            images, labels = omniglot.get_training_batch()
            self.model.train_on_batch([images[:, 0], images[:, 1]], labels)

            if omniglot._epoch_done:
                print('Epoch #' + str(epoch_id) + ' end.')
                epoch_id += 1
                print('Epoch #' + str(epoch_id) + ' start.')
                omniglot._epoch_done = False

                # Decay learning rate by 1% after each epoch
                K.set_value(self.model.optimizer.lr, (K.get_value(self.model.optimizer.lr) * 0.99))

                te_images, te_labels = omniglot.get_random_batch(False)
                ev_images, ev_labels = omniglot.get_random_batch(True)
                
                tr_logs = self.model.test_on_batch([te_images[:, 0], te_images[:, 1]], te_labels)
                ev_logs = self.model.test_on_batch([ev_images[:, 0], ev_images[:, 1]], ev_labels)
                
                tensorboard.on_epoch_end(epoch_id, self.named_logs(tr_logs))
                tensorboard_eval.on_epoch_end(epoch_id, self.named_logs(ev_logs))

        print('Training finished.')
        print('Saving model...')
        model_json = self.model.to_json()
        with open(os.path.join(self.__save_dir, 'model.json'), "w") as json_file:
            json_file.write(model_json)
        
        self.model.save_weights(os.path.join(self.__save_dir, 'model_weights.h5'))
        self.model.save(os.path.join(self.__save_dir, 'model.h5'))

        print('Model saved successfully')

    def test(self, omniglot):
        '''
            Performs testing of the network

            Arguments:
                - omniglot = instance of OmniglotLoader
        '''
        print('Testing started.')
        omniglot._current_alphabet_index = 0
        omniglot.set_training_evaluation_symbols(False)
        omniglot._testing = True

        if not os.path.exists(self.__save_dir):
            os.makedirs(self.__save_dir)

        if not os.path.exists(self.__train_dir):
            os.makedirs(self.__train_dir)

        tensorboard = TensorBoard(log_dir=self.__train_dir, histogram_freq=0, batch_size=20, write_graph=True, write_grads=True)
        tensorboard.set_model(self.model)
        batch_id = 0
        accuracy = []
        print('Testing started.')
        while True:
            if omniglot._evaluation_done:
                break

            images, labels = omniglot.get_test_batch()
            logs = self.model.test_on_batch([images[:, 0], images[:, 1]], labels)
            accuracy.append(logs[1])
            tensorboard.on_test_batch_end(batch_id, self.named_logs(logs))
            batch_id += 1
        
        print('Testing finished.')
        print('Overall accuracy: ' + str(np.mean(accuracy)))

    def get_predictions(self, omniglot):
        '''
            Performs testing of the network

            Arguments:
                - omniglot = instance of OmniglotLoader
        '''
        omniglot._current_alphabet_index = 0
        omniglot.set_training_evaluation_symbols(False)
        labels = []
        predictions = []
        print('Testing started.')
        while True:
            if omniglot._evaluation_done:
                break

            images, true_val = omniglot.get_test_batch()
            pred = self.model.predict_on_batch([images[:, 0], images[:, 1]])
            predictions.append(pred)
            labels.append(true_val)
        print('Testing ended.')
        return predictions, labels

    ####################################################################
    # Methods for getting results for confusion matrix                 #
    ####################################################################
    def test_tn_fp(self, omniglot):
        '''
            Performs testing of true negatives and false positives on the network

            Arguments:
                - omniglot = instance of OmniglotLoader
        '''
        print('Testing false positives and true negatives started.')
        omniglot._current_alphabet_index = 0
        omniglot.set_training_evaluation_symbols(False)
        omniglot._epoch_done = False
        omniglot._testing = True
        if not os.path.exists(self.__confusion_matrix_dir):
            os.makedirs(self.__confusion_matrix_dir)
        
        (false_positives_low, false_positives_high), (true_negatives_low, true_negatives_high) = self._get_empty_arrays_for_each_case()

        y_pred = []
        while True:
            images, _ = omniglot.get_negative_batch(False)
            predictions = self.model.predict_on_batch([images[:, 0], images[:, 1]])
            y_pred.append(predictions)

            self.calculate_high_and_lows(omniglot, predictions, (true_negatives_low, true_negatives_high), (false_positives_low, false_positives_high))

            if omniglot._epoch_done == True:
                break

        # Serialize true positives and false negatives
        np.save(os.path.join(self.__confusion_matrix_dir, 'true_negatives_low'), true_negatives_low)
        np.save(os.path.join(self.__confusion_matrix_dir, 'true_negatives_high'), true_negatives_high)
        np.save(os.path.join(self.__confusion_matrix_dir, 'false_positives_low'), false_positives_low)
        np.save(os.path.join(self.__confusion_matrix_dir, 'false_positives_high'), false_positives_high)

        accuracy = self._compute_accuracy(y_pred)
        print('Testing true negatives and false positives finished.')
        print('Overall accuracy:', str(accuracy))

        return true_negatives_low, true_negatives_high, false_positives_low, false_positives_high

    def test_tp_fn(self, omniglot):
        '''
            Performs testing of true positives and false negatives on the network

            Arguments:
                - omniglot = instance of OmniglotLoader
        '''
        print('Testing true positives and false negatives started.')
        omniglot._current_alphabet_index = 0
        omniglot.set_training_evaluation_symbols(False)
        omniglot._epoch_done = False
        omniglot._testing = True

        if not os.path.exists(self.__confusion_matrix_dir):
            os.makedirs(self.__confusion_matrix_dir)

        # Create arrays for each case, where each digit represents number of samples for given dictionary
        (true_positives_low, true_positives_high), (false_negatives_low, false_negatives_high) = self._get_empty_arrays_for_each_case()

        y_pred = []
        while True:

            images, _ = omniglot.get_positive_batch(False)
            predictions = self.model.predict_on_batch([images[:, 0], images[:, 1]])
            y_pred.append(predictions)

            self.calculate_high_and_lows(omniglot, predictions, (false_negatives_low, false_negatives_high), (true_positives_low, true_positives_high))

            if omniglot._epoch_done == True:
                break

        # Serialize true positives and false negatives
        np.save(os.path.join(self.__confusion_matrix_dir, 'true_positives_low'), true_positives_low)
        np.save(os.path.join(self.__confusion_matrix_dir, 'true_positives_high'), true_positives_high)
        np.save(os.path.join(self.__confusion_matrix_dir, 'false_negatives_low'), false_negatives_low)
        np.save(os.path.join(self.__confusion_matrix_dir, 'false_negatives_high'), false_negatives_high)

        accuracy = self._compute_accuracy(y_pred)
        print('Testing true positives and false negatives finished.')
        print('Overall accuracy: ' + str(accuracy))

        return true_positives_low, true_positives_high, false_negatives_low, false_negatives_high

    def calculate_high_and_lows(self, omniglot, predictions, test_case_0, test_case_1):
        
        for i in range(len(predictions)):
            if predictions[i] < 0.25:
                test_case_0[0][omniglot._current_alphabet_index] += 1
            elif predictions[i] >= 0.25 and predictions[i] < 0.5:
                test_case_0[1][omniglot._current_alphabet_index] += 1
            elif predictions[i] >= 0.5 and predictions[i] < 0.75:
                test_case_1[0][omniglot._current_alphabet_index] += 1
            elif predictions[i] >= 0.75:
                test_case_1[1][omniglot._current_alphabet_index] += 1

    ####################################################################
    # Various public functions                                         #
    ####################################################################
    def named_logs(self, logs):
        '''
            Method that transforms train_on_batch return value to dictionary expected by on_batch_end callback.
            Source: https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8#file-train_on_batch_with_tensorboard-py-L15

            Arguments:
                - logs = logs generated by train_on_batch

            Returns:
                - result = dictionary with logs
        '''
        result = {}
        for l in zip(self.model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    ####################################################################
    # Various private functions                                         #
    ####################################################################
    def _get_tensorboards(self):
        '''
            Create two tensorboards, one to monitor training results, other to monitor evaluation results
            
            Returns:
                tensorboards
        '''

        tensorboard = TensorBoard(
            log_dir=self.__train_dir,
            histogram_freq=0,
            batch_size=20,
            write_graph=True,
            write_grads=True)
        tensorboard.set_model(self.model)

        evaluation_tensorboard = TensorBoard(
            log_dir=self.__test_dir,
            histogram_freq=0,
            batch_size=20,
            write_graph=True,
            write_grads=True)
        evaluation_tensorboard.set_model(self.model)

        return tensorboard, evaluation_tensorboard

    def _get_empty_arrays_for_each_case(self):
        return ([0]*10, [0]*10), ([0]*10, [0]*10)

    def _create_training_directories(self):
        if not os.path.exists(self.__save_dir):
            os.makedirs(self.__save_dir)

        if not os.path.exists(self.__train_dir):
            os.makedirs(self.__train_dir)

        if not os.path.exists(self.__test_dir):
            os.makedirs(self.__test_dir)

    def _compute_accuracy(self, predictions):

        predictions = np.array(predictions)
        pred = predictions.ravel() > 0.5
        pred_sum = np.sum(pred)
        return (pred_sum / len(pred)) * 100
