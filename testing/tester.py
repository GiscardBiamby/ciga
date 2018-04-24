import keras, os
import sklearn as sk, numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold



def test(model, img_dims=(50,50), grayscale=True,  rescale=1.0, shuffle=False, label='age', test_against='ADIENCE'):
    """
    Inputs:
    - model: a pre-trained Keras model
    - img_dims: dimension of image matrices
    - grayscale: 'grayscale' if True else 'rgb'
    - rescale: factor by which to scale the test images
    - shuffle: whether or not the test data should be shuffled
    - label: whether to test on age ('age') or gender ('gender)
    - test_against: either 'IMDB', 'WIKI', or 'ADIENCE' ~ the dataset to test the model with
    """
    if test_against == 'ADIENCE':
        path = '../datasets/processed/adience'
    elif test_against == 'WIKI':
        path = '../datasets/processed/wiki/'
    elif test_against == 'IMDB':
        path = '../datasets/processed/imdb'
    else:
        print("Data set does not exist, please choose from: {'IMDB', 'WIKI', 'ADIENCE'}")
        return

    path += '/age/' if label == 'age' else '/gender/'
    color_mode = 'grayscale' if grayscale else 'rgb'
    test_data_generator = ImageDataGenerator(rescale=rescale)

    test_generator = test_data_generator.flow_from_directory(path,
                                                             color_mode=color_mode,
                                                             target_size=img_dims,
                                                             batch_size=1,
                                                             class_mode='categorical',
                                                             shuffle=shuffle)

    test_score = model.evaluate_generator(test_generator, use_multiprocessing=True)[1]
    return test_score