import os
import random
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import util
import numpy as np
from PIL import Image

class OmniglotLoader():

    def __init__(self, path='./Omniglot', rotation_range=[-10, 10], shear_range=[5.99, 6.57], scale_range=[0.9, 1.2], shift_range=[-2, 2], batch_size=20, use_transformations=True):
        self.path = path
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.batch_size = batch_size
        self.background_dir = 'images_background'
        self.evaluation_dir = 'images_evaluation' #'nn_test'
        self.use_transformations = use_transformations
        self.training_alphabets, self.evaluation_alphabets = self.__create_alphabets()
        self.training_keys, self.testing_keys = self.__split_train_sets()
        self.__current_alphabet_index = 0
        self.__training_alphabet_list = list(self.training_alphabets.keys())
        self.__symbols_list = list(self.training_alphabets[self.__training_alphabet_list[self.__current_alphabet_index]].keys())
        self.__current_symbol_index = 0
        self.__epoch_done = False
        self.__training_alphabet_num = len(self.training_alphabets.keys())
        self.__evaluation_alphabet_list = list(self.evaluation_alphabets.keys())
        self.__evaluation_alphabet_num = len(self.__evaluation_alphabet_list)
        self.__evaluation_done = False

    ####################################################################
    # Getters and setters                                              #
    ####################################################################
    def is_epoch_done(self):
        return self.__epoch_done

    def set_epoch_done(self, epoch_done):
        self.__epoch_done = epoch_done

    def get_current_alphabet_index(self):
        return self.__current_alphabet_index

    def set_current_alphabet_index(self, idx):
        self.__current_alphabet_index = idx

    def set_training_evaluation_symbols(self, training):
        if training:
            self.__symbols_list = list(self.training_alphabets[self.__training_alphabet_list[self.__current_alphabet_index]].keys())
        else:
            #print(type(self.evaluation_alphabets[self.__evaluation_alphabet_list[self.__current_alphabet_index]]))
            #print(self.evaluation_alphabets[self.__evaluation_alphabet_list[self.__current_alphabet_index]])
            #print(self.training_alphabets[self.__training_alphabet_list[self.__current_alphabet_index]])
            self.__symbols_list = list(self.evaluation_alphabets[self.__evaluation_alphabet_list[self.__current_alphabet_index]].keys())

    def is_evaluation_done(self):
        return self.__evaluation_done

    def set_evaluation_done(self, done):
        self.__evaluation_done = done

    def get_current_symbol_idx(self):
        return self.__current_symbol_index

    def get_evaluation_alphabet_names(self):
        return self.__evaluation_alphabet_list

    def is_use_transformations(self):
        return self.use_transformations

    def set_use_transformations(self, use_transformations):
        self.use_transformations = use_transformations

    def __create_alphabets(self):
        
        training_path = os.path.join(self.path, self.background_dir)
        training_alphabets = {}
        for alphabet in os.listdir(training_path):
            
            print('Processing: ' + alphabet)
            characters = {}
            for char in os.listdir(os.path.join(training_path, alphabet)):
                characters[char] = [img for img in os.listdir(os.path.join(training_path, alphabet, char))]
            training_alphabets[alphabet] = characters
        
        evaluation_path = os.path.join(self.path, self.evaluation_dir)
        evaluation_alphabets = {}
        for alphabet in os.listdir(evaluation_path):
            
            print('Processing: ' + alphabet)
            characters = {}
            for char in os.listdir(os.path.join(evaluation_path, alphabet)):
                characters[char] = [img for img in os.listdir(os.path.join(evaluation_path, alphabet, char))]
            
            evaluation_alphabets[alphabet] = characters
        
        return (training_alphabets, evaluation_alphabets)

    def __split_train_sets(self):
        """
            Perform training and testing split at 80% - 20%"
        """
        
        alphabet_indices = list(range(len(self.training_alphabets)))
        training_indices = random.sample(range(0, len(self.training_alphabets) - 1), k=int(0.8*len(self.training_alphabets)))
        testing_indices = set(alphabet_indices) - set(training_indices)

        training_keys = [list(self.training_alphabets.keys())[i] for i in training_indices]
        testing_keys = [list(self.training_alphabets.keys())[i] for i in testing_indices]
        return (training_keys, testing_keys)

    def create_training_pairs(self):
        #training_path = os.path.join(self.path, )
        alphabets = {}
        for alphabet in self.training_keys:
            #alphabet_path == os.path.join(training_path, alphabet)
            symbol_dict = {}
            for symbol in self.training_alphabets[alphabet]:
                symbol_paths = [os.path.join(self.path, 'images_background', alphabet, symbol, img) for img in os.listdir(os.path.join(self.path, 'images_background', alphabet, symbol))]
                symbol_dict[symbol] = [self.__get_image(img_path) for img_path in symbol_paths]
                
            alphabets[alphabet] = symbol_dict
        #print(alphabets[self.training_keys[0]]['character01'])
        self.image_alphabets = alphabets
        pairs = []
        labels = []
        for alphabet in self.image_alphabets:
            for symbol in self.image_alphabets[alphabet].keys():
                for img in self.image_alphabets[alphabet][symbol]:
                    same_imgs = random.sample(range(0, 20), 10)
                    
                    test_case = img
                    for i in same_imgs:
                        image = alphabets[alphabet][symbol][i]
                        if self.use_transformations:
                            image = self.__transform_image(image)
                        pairs += [[test_case, image]]

                        #random_img = self.__get_random_image(alphabet)
                        #pairs += [[test_case, random_img]]
                        labels += [1, 0]

    def create_pairs(self, directory, alphabets, keys):
        image_alphabets = {}
        for alphabet in keys:
            symbol_dict = {}
            for symbol in alphabets[alphabet]:
                symbol_paths = [os.path.join(self.path, directory, alphabet, symbol, img) for img in os.listdir(os.path.join(self.path, directory, alphabet, symbol))]
                symbol_dict[symbol] = [self.__get_image(img_path) for img_path in symbol_paths]
                
            image_alphabets[alphabet] = symbol_dict
        
        pairs = []
        labels = []
        for alphabet in image_alphabets:
            for symbol in image_alphabets[alphabet].keys():
                for img in image_alphabets[alphabet][symbol]:
                    same_imgs = random.sample(range(0, 20), 10)
                    
                    test_case = img
                    for i in same_imgs:
                        image = image_alphabets[alphabet][symbol][i]
                        if self.use_transformations:
                            image = self.__transform_image(image)
                        pairs += [[test_case, image]]

                        random_img = self.__get_random_image(alphabet, image_alphabets, self.background_dir)
                        pairs += [[test_case, random_img]]
                        labels += [1, 0]

        return np.array(pairs), np.array(labels)

    def __get_random_image(self, current_alphabet, alphabet_dict, directory):
        """
            Gets a random image of a random symbol from a random alphabet.
            Current alphabet is excluded.
        """
        keys = set(alphabet_dict.keys())
        keys.remove(current_alphabet)
        alphabet_sample = keys
        random_alphabet = random.choice(list(alphabet_sample))
        random_symbol = random.choice(list(alphabet_dict[random_alphabet]))
        random_idx = random.randint(0, 19)

        random_img = self.__get_image(os.path.join(self.path, directory, random_alphabet, random_symbol, alphabet_dict[random_alphabet][random_symbol][random_idx]))
        if self.use_transformations:
            random_img = self.__transform_image(random_img)
        return random_img


    def get_training_batch(self):
        directory = self.background_dir        
        pairs = []
        labels = []

        images = random.sample(range(0, 20), 11)
        current_alphabet = self.__training_alphabet_list[self.__current_alphabet_index]
        current_symbol = self.__symbols_list[self.__current_symbol_index]
        current_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, self.training_alphabets[current_alphabet][current_symbol][0]))
        '''
        print('============================================================')
        print('Current alphabet: ' + current_alphabet)
        print('Current symbol: ' + current_symbol)
        print('Current symbol: ' + str(self.__current_symbol_index))
        '''
        for img in images[1:]:
            second_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, self.training_alphabets[current_alphabet][current_symbol][img]))
            
            if self.use_transformations:
                second_image = self.__transform_image(second_image)
            pairs += [[current_image, second_image]]

            random_img = self.__get_random_image(current_alphabet, self.training_alphabets, self.background_dir)
            pairs += [[current_image, random_img]]
            labels += [1, 0]

        self.__current_symbol_index += 1
        if self.__current_symbol_index == len(self.training_alphabets[current_alphabet]):
            self.__current_symbol_index = 0
            self.__current_alphabet_index += 1
            print(str(round((self.__current_alphabet_index / self.__training_alphabet_num) * 100.00, 2)) + ' %% of alphabets done')
            if self.__current_alphabet_index == len(self.training_alphabets):
                self.__current_alphabet_index = 0
                self.__epoch_done = True
            self.__symbols_list = list(self.training_alphabets[self.__training_alphabet_list[self.__current_alphabet_index]].keys())
                
        pairs = np.array(pairs)
        return pairs, np.array(labels)

    def __transform_image(self, img):
        theta = 0
        dx, dy = 0, 0
        sx, sy = 1, 1
        shear_factor = 0
        # Calculating transformation
        if np.random.uniform(low=0, high=1) < 0.5:
            theta = np.random.uniform(low=self.rotation_range[0], high=self.rotation_range[1])
        if np.random.uniform(low=0, high=1) < 0.5:
            dx = np.random.uniform(low=self.shift_range[0], high=self.shift_range[1])
        if np.random.uniform(low=0, high=1) < 0.5:
            dy = np.random.uniform(low=self.shift_range[0], high=self.shift_range[1])
        if np.random.uniform(low=0, high=1) < 0.5:
            sx = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        if np.random.uniform(low=0, high=1) < 0.5:
            sy = np.random.uniform(low=self.scale_range[0], high=self.scale_range[1])
        if np.random.uniform(low=0, high=1) < 0.5:
            shear_factor = np.random.uniform(low=self.shear_range[0], high=self.shear_range[1])

        transform_map = AffineTransform(scale=(sx, sy), rotation=np.deg2rad(theta), shear=shear_factor, translation=(dx, dy))
        transformed = warp(img, inverse_map=transform_map, preserve_range=True)
        '''
        plt.figure(0, figsize=(8, 3))
        plt.axis('off')
        plt.imshow(transformed, cmap='gray')
        plt.show()
        '''
        return transformed

    def __get_image(self, path):
        img = plt.imread(path)
        img = img.reshape(105, 105, 1)
        # Invert image to be easier to use with affine transformations
        inverted = util.invert(img)
        return inverted

    def get_random_batch(self, directory, test_batch):
        pairs = []
        labels = []

        images = random.sample(range(0, 20), 11)
        if test_batch:
            random_alphabet_idx = random.randint(0, len(self.__evaluation_alphabet_list) - 1)
            random_alphabet = self.__evaluation_alphabet_list[random_alphabet_idx]
            symbols_list = list(self.evaluation_alphabets[random_alphabet].keys())
            random_symbol_idx = random.randint(0, len(symbols_list) - 1)
            random_symbol = symbols_list[random_symbol_idx]
            current_image = self.__get_image(os.path.join(self.path, directory, random_alphabet, random_symbol, self.evaluation_alphabets[random_alphabet][random_symbol][0]))
        else:
            random_alphabet_idx = random.randint(0, len(self.__training_alphabet_list) - 1)
            random_alphabet = self.__training_alphabet_list[random_alphabet_idx]
            symbols_list = list(self.training_alphabets[random_alphabet].keys())
            random_symbol_idx = random.randint(0, len(symbols_list) - 1)
            random_symbol = symbols_list[random_symbol_idx]
            current_image = self.__get_image(os.path.join(self.path, directory, random_alphabet, random_symbol, self.training_alphabets[random_alphabet][random_symbol][0]))
        
        for img in images[1:]:
            if test_batch:
                second_image = self.__get_image(os.path.join(self.path, directory, random_alphabet, random_symbol, self.evaluation_alphabets[random_alphabet][random_symbol][img]))
            else:
                second_image = self.__get_image(os.path.join(self.path, directory, random_alphabet, random_symbol, self.training_alphabets[random_alphabet][random_symbol][img]))
            
            if self.use_transformations:
                second_image = self.__transform_image(second_image)
            pairs += [[current_image, second_image]]

            if test_batch:
                random_img = self.__get_random_image(random_alphabet, self.evaluation_alphabets, self.evaluation_dir)
            else:
                random_img = self.__get_random_image(random_alphabet, self.training_alphabets, self.background_dir)
            pairs += [[current_image, random_img]]
            labels += [1, 0]

        return np.array(pairs), np.array(labels)

    def get_positive_batch(self, training):
        pairs = []
        labels = []

        random_pairs = [(random.randint(0, self.batch_size - 1), random.randint(0, self.batch_size - 1)) for k in range(self.batch_size)]
        if training:
            current_alphabet = self.__training_alphabet_list[self.__current_alphabet_index]
            current_symbol = self.__symbols_list[self.__current_symbol_index]
            directory = self.background_dir
            img_names = self.training_alphabets[current_alphabet][current_symbol]
        else:
            current_alphabet = self.__evaluation_alphabet_list[self.__current_alphabet_index]
            current_symbol = self.__symbols_list[self.__current_symbol_index]
            directory = self.evaluation_dir
            img_names = self.evaluation_alphabets[current_alphabet][current_symbol]

        for pair in random_pairs:
            left_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, img_names[pair[0]]))
            right_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, img_names[pair[1]]))

            if self.use_transformations:
                left_image = self.__transform_image(left_image)
                right_image = self.__transform_image(right_image)
            
            pairs += [[left_image, right_image]]
            labels += [1]

        self.__change_symbol(training)

        return np.array(pairs), np.array(labels)

    def get_negative_batch(self, training):
        pairs = []
        labels = []

        image_idx = random.randint(0, self.batch_size - 1)
        if training:
            current_alphabet = self.__training_alphabet_list[self.__current_alphabet_index]
            current_symbol = self.__symbols_list[self.__current_symbol_index]
            directory = self.background_dir
            img_names = self.training_alphabets[current_alphabet][current_symbol]
            alphabet_dict = self.training_alphabets
        else:
            current_alphabet = self.__evaluation_alphabet_list[self.__current_alphabet_index]
            current_symbol = self.__symbols_list[self.__current_symbol_index]
            directory = self.evaluation_dir
            img_names = self.evaluation_alphabets[current_alphabet][current_symbol]
            alphabet_dict = self.evaluation_alphabets

        for _ in range(0, self.batch_size - 1):
            left_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, img_names[image_idx]))
            right_image = self.__get_random_image(current_alphabet, alphabet_dict, directory)

            if self.use_transformations:
                left_image = self.__transform_image(left_image)
                right_image = self.__transform_image(right_image)
            
            pairs += [[left_image, right_image]]
            labels += [0]

        self.__change_symbol(training)

        return np.array(pairs), np.array(labels)

    def __change_symbol(self, training):
        self.__current_symbol_index += 1
        if training:
            current_alphabet = self.__training_alphabet_list[self.__current_alphabet_index]
            if self.__current_symbol_index == len(self.training_alphabets[current_alphabet]):
                self.__current_symbol_index = 0
                self.__current_alphabet_index += 1
                print(str(round((self.__current_alphabet_index / self.__training_alphabet_num) * 100.00, 2)) + ' %% of alphabets done')
                if self.__current_alphabet_index == len(self.training_alphabets):
                    self.__current_alphabet_index = 0
                    self.__epoch_done = True
                self.__symbols_list = list(self.training_alphabets[self.__training_alphabet_list[self.__current_alphabet_index]].keys())
        else:
            current_alphabet = self.__evaluation_alphabet_list[self.__current_alphabet_index]
            if self.__current_symbol_index == len(self.evaluation_alphabets[current_alphabet]):
                self.__current_symbol_index = 0
                self.__current_alphabet_index += 1
                print(str(round((self.__current_alphabet_index / self.__evaluation_alphabet_num) * 100.00, 2)) + ' %% of alphabets done')
                if self.__current_alphabet_index == len(self.evaluation_alphabets):
                    self.__current_alphabet_index = 0
                    self.__epoch_done = True
                self.__symbols_list = list(self.evaluation_alphabets[self.__evaluation_alphabet_list[self.__current_alphabet_index]].keys())

    def get_test_batch(self):
        directory = self.evaluation_dir
        pairs = []
        labels = []

        images = random.sample(range(0, 20), 11)
        current_alphabet = self.__evaluation_alphabet_list[self.__current_alphabet_index]
        current_symbol = self.__symbols_list[self.__current_symbol_index]
        current_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, self.evaluation_alphabets[current_alphabet][current_symbol][0]))

        print('============================================================')
        print('Current alphabet: ' + current_alphabet)
        print('Current symbol: ' + current_symbol)
        print('Current symbol: ' + str(self.__current_symbol_index))

        for img in images[1:]:
            second_image = self.__get_image(os.path.join(self.path, directory, current_alphabet, current_symbol, self.evaluation_alphabets[current_alphabet][current_symbol][img]))
            
            if self.use_transformations:
                second_image = self.__transform_image(second_image)
            pairs += [[current_image, second_image]]

            random_img = self.__get_random_image(current_alphabet, self.evaluation_alphabets, self.evaluation_dir)
            pairs += [[current_image, random_img]]
            labels += [1, 0]

        self.__current_symbol_index += 1
        if self.__current_symbol_index == len(self.evaluation_alphabets[current_alphabet]):
            self.__current_symbol_index = 0
            self.__current_alphabet_index += 1
            print(str(round((self.__current_alphabet_index / self.__evaluation_alphabet_num) * 100.0, 2)) + ' % of alphabets done')
            if self.__current_alphabet_index == len(self.evaluation_alphabets):
                self.__current_alphabet_index = 0
                self.__evaluation_done = True
            self.__symbols_list = list(self.evaluation_alphabets[self.__evaluation_alphabet_list[self.__current_alphabet_index]].keys())
        
        pairs = np.array(pairs)
        return pairs, np.array(labels)