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
import keras
from keras import optimizers
from keras import backend as K
from testing.confusion_mtx import *


def off_by_one_categorical_accuracy(y_true, y_pred):
    num_classes = K.int_shape(y_pred)[1]

    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    shifted_up = K.clip(y_true + 1, 0, num_classes)
    shifted_down = K.clip(y_true - 1, 0, num_classes)

    comparison1 = tf.equal(y_true, y_pred)
    comparison2 = tf.equal(shifted_up, y_pred)
    comparison3 = tf.equal(shifted_down, y_pred)

    mask = tf.logical_or(comparison1, comparison2)
    mask = tf.logical_or(mask, comparison3)
    return K.cast(mask, K.floatx())

def k_fold_cross_val(dataset,
                     k,
                     label_type,
                     model_type,
                     trainer_config,
                     epochs=1,
                     img_size=50,
                     batch_size=32,
                     grayscale=True,
                     weights=None,
                     architecture=None,
                     model=None):

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

    # Build or load k Models:
    if model:
        models = [keras.models.load_model(model, custom_objects={'off_by_one_categorical_accuracy': off_by_one_categorical_accuracy}) for _ in range(k)]
    else:
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
    confusion_mtx(models, validation_generators, model_name=model_type, k=k)

def main():
    parser = argparse.ArgumentParser(
        description='File to complete k folds cross validation on input dataset (potentially with pretrained weights).')
    parser.add_argument('dataset', help='Dataset to use, either "adience", "imdb" or "wiki"')
    parser.add_argument('k', type=int, help='The number of folds to use, typically k=5 or k=7')
    parser.add_argument('label_type', help="The label type to be used, either: 'age', 'age_gender' or 'gender'")
    parser.add_argument('model_type', help="The model type to be used, either: 'vgg-16', 'andreynet' or 'resnet'")
    parser.add_argument('-w', '--weights', help="File path to pretrained weights.")
    parser.add_argument('-p', '--pre_trained_model', help="File path to a pretrained model.")
    args = parser.parse_args()
    grayscale = True
    img_size = 224
    batch_size = 32
    epochs = 1

    # (IF MODEL == RESNET || MODEL == ANDREYNET) && (not using pre-trained model):
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
    if args.pre_trained_model:
        k_fold_cross_val(args.dataset,
                         args.k,
                         args.label_type,
                         args.model_type,
                         trainer_config,
                         epochs=epochs,
                         img_size=img_size,
                         batch_size=batch_size,
                         grayscale=grayscale,
                         architecture=architecture,
                         model=args.pre_trained_model)
    elif args.weights:
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

