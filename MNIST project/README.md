# Experiment with MNIST dataset and Keras' example

The goal of this experiment was to test how different affine transformations affect the training and testing of siamese neural networks.
Implementation is done in Tensorflow and Keras.

Four cases are covered: 
1. training and testing without transformations,
2. training and testing with transformations,
3. training without transformations and testing with transformations,
4. training with transformations and testing without transformations 

To train models with and without transformations, run **train_models.py**.

To determine positive and negative pairs, run **identifying_images.py**. It is necessary to set up **model_location** variable to point to the directory that contains models trained with **train_models.py** ('./data/...')

To get results organized in table, run **get_tables.py**.

Because during the analysis of the results the highest difference in results was observed for digit 4, these results were analyzed in more detail. To get embedding for digit 4, it is necessary to run **projecting_4.py** and **get_plots.py**, to get plots with marked positive and negative results.

Table overview of the results is shown in **tables.pdf**.
