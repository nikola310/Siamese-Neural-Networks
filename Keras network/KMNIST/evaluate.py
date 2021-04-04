from tensorflow.keras.models import load_model
from train_models import contrastive_loss, compute_final_accuracy, prepare_data_for_testing, compute_final_accuracy, prepare_data_for_training

loc = '/home/nikola/Programming/kmnist2/models/2019-12-12 22-46-00/siamese_model_transformations.h5'

model = load_model(loc, custom_objects={'contrastive_loss': contrastive_loss})

(te_pairs, te_y) = prepare_data_for_testing(transformations=True)
#(te_pairs, te_y) = prepare_data_for_training(transformations=True)
compute_final_accuracy(model, None, None, te_pairs, te_y)
