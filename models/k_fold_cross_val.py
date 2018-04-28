import sys
sys.path.append('../')
import os
import numpy as np
import argparse
import models.vgg as cigaModels
import training.trainer as cigaTraining
from keras import optimizers
from testing.confusion_mtx import *


def k_fold_cross_val(dataset, k, label_type, trainer_config, epochs=1, img_size=50, batch_size=32, grayscale=True, weights=None):
    num_channels = 1 if grayscale else 3
    img_shape = (img_size, img_size, num_channels)
    fold_paths = "../datasets/processed/" + dataset + "/" + label_type + "_fold_"
    fold_paths = [fold_paths + str(i) + "/" for i in range(1, k+1)]

    # Build k data generators:
    train_generators, validation_generators = [], []
    for i in range(k):
        train, valid = cigaTraining.generatorsBuilder(
            fold_paths[i], img_dims=(img_size, img_size), batch_size=batch_size, grayscale=grayscale)
        train_generators += [train]
        validation_generators += [valid]
    num_classes = train.num_classes

    # Build k Models:
    models = [cigaModels.getModel_vgg16(img_shape, num_classes) for _ in range(k)]

    # Load pretrained weights:
    if weights:
        for i in range(k):
            models[i].load_weights(weights, by_name=False)

    # Train k models:
    trainers = [cigaTraining.BasicTrainer(model=models[i], config=trainer_config, enable_sms=False) for i in range(k)]
    models = [trainers[i].train(validation_generators[i], train_generators[i], batch_size=batch_size, epochs=epochs) for i in range(k)]

    # Save k models:
    for i in range(k):
        trainers[i].saveModel('VGG-16 fold %d validation accuracy ' % (i+1) + str(max(models[i].history.history['val_acc'])))

    # Generate k+1 confusion matrices (+1 for matrix across all folds):
    confusion_mtx(models, validation_generators, model_name="VGG-16", k=k)

def main():
    parser = argparse.ArgumentParser(
        description='File to complete k folds cross validation on input dataset (potentially with pretrained weights).')
    parser.add_argument('dataset', help='Dataset to use, either "adience", "imdb" or "wiki"')
    parser.add_argument('k', type=int, help='The number of folds to use, typically k=5 or k=7')
    parser.add_argument('label_type', help="The label type to be used, either: 'age', 'age_gender' or 'gender'")
    parser.add_argument('-w', '--weights', help="File path to pretrained weights.")
    args = parser.parse_args()
    grayscale = True
    img_size = 224
    batch_size = 128
    epochs = 20
    trainer_config = {
        "reduce_lr_params": {
            "factor": 0.2
            , "patience": 1
            , "min_lr": 1e-8
            , "verbose": 1
        }
        , "optimizer": "sgd"
        , "optimizer_params": {"lr": 1e-2, "decay": 5e-4, "momentum": 0.9, "nesterov": True}
    }
    if args.weights:
        k_fold_cross_val(args.dataset,
                         args.k,
                         args.label_type,
                         trainer_config,
                         epochs=epochs,
                         img_size=img_size,
                         batch_size=batch_size,
                         grayscale=grayscale,
                         weights=args.weights)
    else:
        k_fold_cross_val(args.dataset,
                         args.k,
                         args.label_type,
                         trainer_config,
                         epochs=epochs,
                         img_size=img_size,
                         batch_size=batch_size,
                         grayscale=grayscale)

if __name__ == '__main__':
    main()

# class MyDataGenerator(ImageDataGenerator):
#     def __init__(self,
#                  featurewise_center=False,
#                  samplewise_center=False,
#                  featurewise_std_normalization=False,
#                  samplewise_std_normalization=False,
#                  zca_whitening=False,
#                  zca_epsilon=1e-6,
#                  rotation_range=0.,
#                  width_shift_range=0.,
#                  height_shift_range=0.,
#                  brightness_range=None,
#                  shear_range=0.,
#                  zoom_range=0.,
#                  channel_shift_range=0.,
#                  fill_mode='nearest',
#                  cval=0.,
#                  horizontal_flip=False,
#                  vertical_flip=False,
#                  rescale=None,
#                  preprocessing_function=None,
#                  data_format=None,
#                  validation_split=0.0,
#                  split=(0,0)):
#         super(MyDataGenerator, self).__init__(self,
#                  featurewise_center=False,
#                  samplewise_center=False,
#                  featurewise_std_normalization=False,
#                  samplewise_std_normalization=False,
#                  zca_whitening=False,
#                  zca_epsilon=1e-6,
#                  rotation_range=0.,
#                  width_shift_range=0.,
#                  height_shift_range=0.,
#                  brightness_range=None,
#                  shear_range=0.,
#                  zoom_range=0.,
#                  channel_shift_range=0.,
#                  fill_mode='nearest',
#                  cval=0.,
#                  horizontal_flip=False,
#                  vertical_flip=False,
#                  rescale=None,
#                  preprocessing_function=None,
#                  data_format=None,
#                  validation_split=0.0)
#         self._validation_split = split
#
# class MyDirectoryIterator(DirectoryIterator):
#     def __init__(self, directory, image_data_generator,
#                  target_size=(256, 256), color_mode='rgb',
#                  classes=None, class_mode='categorical',
#                  batch_size=32, shuffle=True, seed=None,
#                  data_format=None,
#                  save_to_dir=None, save_prefix='', save_format='png',
#                  follow_links=False,
#                  subset=None,
#                  interpolation='nearest'):
#         if data_format is None:
#             data_format = K.image_data_format()
#         self.directory = directory
#         self.image_data_generator = image_data_generator
#         self.target_size = tuple(target_size)
#         if color_mode not in {'rgb', 'grayscale'}:
#             raise ValueError('Invalid color mode:', color_mode,
#                              '; expected "rgb" or "grayscale".')
#         self.color_mode = color_mode
#         self.data_format = data_format
#         if self.color_mode == 'rgb':
#             if self.data_format == 'channels_last':
#                 self.image_shape = self.target_size + (3,)
#             else:
#                 self.image_shape = (3,) + self.target_size
#         else:
#             if self.data_format == 'channels_last':
#                 self.image_shape = self.target_size + (1,)
#             else:
#                 self.image_shape = (1,) + self.target_size
#         self.classes = classes
#         if class_mode not in {'categorical', 'binary', 'sparse',
#                               'input', None}:
#             raise ValueError('Invalid class_mode:', class_mode,
#                              '; expected one of "categorical", '
#                              '"binary", "sparse", "input"'
#                              ' or None.')
#         self.class_mode = class_mode
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
#         self.interpolation = interpolation
#
#         if subset is not None:
#             validation_split = self.image_data_generator._validation_split
#             if subset == 'validation':
#                 split = validation_split
#             elif subset == 'training':
#                 split = (validation_split, 1)
#             else:
#                 raise ValueError('Invalid subset name: ', subset,
#                                  '; expected "training" or "validation"')
#         else:
#             split = None
#         self.subset = subset
#
#         white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}
#
#         # first, count the number of samples and classes
#         self.samples = 0
#
#         if not classes:
#             classes = []
#             for subdir in sorted(os.listdir(directory)):
#                 if os.path.isdir(os.path.join(directory, subdir)):
#                     classes.append(subdir)
#         self.num_classes = len(classes)
#         self.class_indices = dict(zip(classes, range(len(classes))))
#
#         pool = multiprocessing.pool.ThreadPool()
#         function_partial = partial(_count_valid_files_in_directory,
#                                    white_list_formats=white_list_formats,
#                                    follow_links=follow_links,
#                                    split=split)
#         self.samples = sum(pool.map(function_partial,
#                                     (os.path.join(directory, subdir)
#                                      for subdir in classes)))
#
#         print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))
#
#         # second, build an index of the images in the different class subfolders
#         results = []
#
#         self.filenames = []
#         self.classes = np.zeros((self.samples,), dtype='int32')
#         i = 0
#         for dirpath in (os.path.join(directory, subdir) for subdir in classes):
#             results.append(pool.apply_async(_list_valid_filenames_in_directory,
#                                             (dirpath, white_list_formats, split,
#                                              self.class_indices, follow_links)))
#         for res in results:
#             classes, filenames = res.get()
#             self.classes[i:i + len(classes)] = classes
#             self.filenames += filenames
#             i += len(classes)
#
#         pool.close()
#         pool.join()
#         super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)
#
#
#
# def k_fold_cross_val(k, model, epochs, img_dims=(50, 50), batch_size=32, grayscale=True, label='age', dataset='ADIENCE'):
#     """
#     Inputs:
#     - k: number of folds
#     - model: a keras model (not pre-compiled)
#     - epochs: self explanatory
#     - img_dims: dimension of image matrices
#     - batch_size: self explanatory
#     - grayscale: self explanatory
#     - label: whether to train on age ('age') or gender ('gender)
#     - dataset: either 'IMDB', 'WIKI', or 'ADIENCE' ~ the dataset to use cross validation on
#     """
#     if dataset == 'ADIENCE':
#         path = '../datasets/processed/adience'
#     elif dataset == 'WIKI':
#         path = '../datasets/processed/wiki/'
#     elif dataset == 'IMDB':
#         path = '../datasets/processed/imdb'
#     else:
#         print("Inproper data set name!")
#     path += '/age/' if label == 'age' else '/gender/'
#
#     # train_data_dir = path + 'train/'
#     # valid_data_dir = path + 'valid/'
#
#     color_mode = 'grayscale' if grayscale else 'rgb'
#
#     dataset_datagen = ImageDataGenerator(samplewise_center=True,
#                                          samplewise_std_normalization=True,
#                                          shear_range=0.2,
#                                          zoom_range=0.2,
#                                          horizontal_flip=True,
#                                          validation_split=1.0/k)
#
#     dataset_generator = dataset_datagen.flow_from_directory(path,
#                                                             color_mode=color_mode,
#                                                             target_size=img_dims,
#                                                             batch_size=batch_size,
#                                                             class_mode='categorical')
#
#
#     valid_datagen = ImageDataGenerator(samplewise_center=True,
#                                        samplewise_std_normalization=True)
#
#     valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
#                                                         color_mode=color_mode,
#                                                         target_size=img_dims,
#                                                         batch_size=batch_size,
#                                                         class_mode='categorical')
#
#     model.fit(X[train], Y[train], epochs=150, batch_size=k, verbose=0)