import keras, os
import sklearn as sk, numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold



def k_fold_cross_val(k, model, epochs, img_dims=(50, 50), batch_size=32, grayscale=True, label='age', dataset='ADIENCE'):
    """
    Inputs:
    - k: number of folds
    - model: a keras model (not pre-compiled)
    - epochs: self explanatory
    - img_dims: dimension of image matrices
    - batch_size: self explanatory
    - grayscale: self explanatory
    - label: whether to train on age ('age') or gender ('gender)
    - dataset: either 'IMDB', 'WIKI', or 'ADIENCE' ~ the dataset to use cross validation on
    """
    if dataset == 'ADIENCE':
        path = '../datasets/processed/adience'
    elif dataset == 'WIKI':
        path = '../datasets/processed/wiki/'
    elif dataset == 'IMDB':
        path = '../datasets/processed/imdb'
    else:
        print("Inproper data set name!")
    path += '/age/' if label == 'age' else '/gender/'

    # train_data_dir = path + 'train/'
    # valid_data_dir = path + 'valid/'

    color_mode = 'grayscale' if grayscale else 'rgb'

    dataset_datagen = ImageDataGenerator(samplewise_center=True,
                                         samplewise_std_normalization=True,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         validation_split=1.0/k)

    dataset_generator = dataset_datagen.flow_from_directory(path,
                                                            color_mode=color_mode,
                                                            target_size=img_dims,
                                                            batch_size=batch_size,
                                                            class_mode='categorical')


    valid_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True)

    valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
                                                        color_mode=color_mode,
                                                        target_size=img_dims,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    model.fit(X[train], Y[train], epochs=150, batch_size=k, verbose=0)


def adience_crossval_score(model, epochs, img_dims=(224, 224), batch_size=32, grayscale=True, label_by="age"):
    """
    Performs k-fold crossvalidation on the pre-defined splits on the adience dataset.
    :param model:
    :param epochs:
    :param img_dims:
    :param batch_size:
    :param grayscale:
    :param label_by:
    :return: Average accuracy, average one-off accuracy, std.dev of accuracy
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets/processed/adience")