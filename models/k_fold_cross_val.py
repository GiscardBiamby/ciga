import sys
sys.path.append('../')
import os
import numpy as np
import argparse
import models.vgg as vgg
import models.resnet as resnet
import models.AndreyNet as andreynet
import training.trainer as cigaTraining
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from testing.confusion_mtx import *

def k_fold_cross_val(dataset, k, label_type, model_type, trainer_config, epochs=1, img_size=50, batch_size=32, grayscale=True, weights=None, architecture=None):
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
    models = []
    if model_type == 'vgg-16':
        models = [vgg.getModel_vgg16(img_shape, num_classes) for _ in range(k)]
    elif model_type == 'andreynet':
        models = [andreynet.AndreyNet(architecture, img_shape, num_classes) for _ in range(k)]
    elif model_type == 'resnet':
        models = [resnet.resnetBuilder(architecture, img_shape, num_classes) for _ in range(k)]

    # Load pretrained weights:
    if weights:
        for i in range(k):
            models[i].load_weights(weights, by_name=False)

    # Train k models:
    trainers = [cigaTraining.BasicTrainer(model=models[i], config=trainer_config, enable_sms=False) for i in range(k)]
    models = [trainers[i].train(validation_generators[i], train_generators[i], batch_size=batch_size, epochs=epochs) for i in range(k)]

    # Save k models:
    for i in range(k):
        trainers[i].saveModel(model_type + ' fold %d validation accuracy ' % (i+1) + str(max(models[i].history.history['val_acc'])))

    # Generate k+1 confusion matrices (+1 for matrix across all folds):
    confusion_mtx(models, validation_generators, model_name="VGG-16", k=k)

def main():
    parser = argparse.ArgumentParser(
        description='File to complete k folds cross validation on input dataset (potentially with pretrained weights).')
    parser.add_argument('dataset', help='Dataset to use, either "adience", "imdb" or "wiki"')
    parser.add_argument('k', type=int, help='The number of folds to use, typically k=5 or k=7')
    parser.add_argument('label_type', help="The label type to be used, either: 'age', 'age_gender' or 'gender'")
    parser.add_argument('model_type', help="The model type to be used, either: 'vgg-16', 'andreynet' or 'resnet'")
    parser.add_argument('-w', '--weights', help="File path to pretrained weights.")
    args = parser.parse_args()
    grayscale = True
    img_size = 224
    batch_size = 16
    epochs = 1

    # IF MODEL == RESNET || MODEL == ANDREYNET:
    # Change values in this section for ResNet or AndreyNet architecture variable:
    architecture, layers, stages, dense_size, dense = {}, 4, 5, 512, 1
    architecture['stages'] = [[layers, 64 * (2 ** i)] for i in range(stages)]
    architecture['dense'] = [dense_size for _ in range(dense)]

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
                         args.model_type,
                         trainer_config,
                         epochs=epochs,
                         img_size=img_size,
                         batch_size=batch_size,
                         grayscale=grayscale,
                         weights=args.weights,
                         architecture=architecture)
    else:
        k_fold_cross_val(args.dataset,
                         args.k,
                         args.label_type,
                         args.model_type,
                         trainer_config,
                         epochs=epochs,
                         img_size=img_size,
                         batch_size=batch_size,
                         grayscale=grayscale,
                         architecture=architecture)

if __name__ == '__main__':
    main()

