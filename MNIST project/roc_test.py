import scikitplot as skplt
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from train_models import create_pairs, contrastive_loss

shape = (28, 28)
num_classes = 10
rotation_range = [-10, 10]
shear_range = [-12, 12]
scale_range = [0.9, 1.2]
shift_range = [-2, 2]

model_location = './test_models/'
plot_location_tp_fn = './figures_tp_fn/'
digits_location_tp_fn = './digits_tp_fn/'
plot_location_tn_fp = './figures_tn_fp/'
digits_location_tn_fp = './digits_tn_fp/'

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    input_shape = x_train.shape[1:]

    model = load_model(join(model_location, 'siamese_model_transformations.h5'), custom_objects={'contrastive_loss': contrastive_loss})
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    labels = []
    transformations=True
    te_pairs, te_y = create_pairs(x_test, digit_indices, labels, transform=False)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    fpr, tpr, thresholds = skplt.metrics.roc_curve(te_y, y_pred, pos_label=0)
    plt.plot(fpr,tpr)
    plt.show()