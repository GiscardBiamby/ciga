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
import copy

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
    print('+====================================================================+')
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
        models = []
        for _ in range(k):
            print("Loading weights: '{}'".format(model))
            models.append(keras.models.load_model(
                model
                , custom_objects={'off_by_one_categorical_accuracy': cigaTraining.off_by_one_categorical_accuracy})
            )
    else:
        models = []
        if model_type == 'vgg-16':
            models = [vgg.getModel_vgg16(img_shape, num_classes) for _ in range(k)]
        elif model_type == 'sameres':
            models = [andreynet.AndreyNet(architecture, img_shape, num_classes) for _ in range(k)]
        elif model_type == 'resnet':
            models = [resnet.resnetBuilder(architecture, img_shape, num_classes) for _ in range(k)]

        # Load pretrained weights:
        if weights:
            for i in range(k):
                print("Loading weights: '{}'".format(weights))
                models[i].load_weights(weights, by_name=False)

    # Var to store val_acc and off-by-one-acc:
    results = np.zeros((k,4))

    # Train k models:
    trainers = []
    for i in range(k):
        trainer_config_i = copy.deepcopy(trainer_config)
        trainer_config_i["fold_num"] = i
        trainer_config_i["kfolds_k"] = k
        trainers.append(cigaTraining.BasicTrainer(model=models[i], config=trainer_config_i,
                                                 enable_sms=False))
        print("Training fold {}/{}...".format(i+1, k))
        try:
            trainers[i].train(
                validation_generators[i]
                , train_generators[i]
                , batch_size=batch_size
                , epochs=epochs
            )
        except Exception as e:
            print("ERROR while training model w/ config: ", trainer_config)
            print("Error: ", e)
            # Save to prevent trying to re-run same config again:
            trainers[i].saveModel(
                'kfolds-{}-{}-fold{}-val_acc-ERROR'.format(
                    model_type
                    , label_type
                    , i + 1
                )
                , saveConfigOnly=True
            )
            continue

        # Capture curr. fold results:
        best_index = np.argmax(np.array(models[i].history.history['val_acc']), axis=0)
        results[i,0] = models[i].history.history['acc'][best_index]
        if "off_by_one_categorical_accuracy" in models[i].history.history:
            results[i, 1] = models[i].history.history['off_by_one_categorical_accuracy'][best_index]
        results[i,2] = models[i].history.history['val_acc'][best_index]
        if "val_off_by_one_categorical_accuracy" in models[i].history.history:
            results[i, 3] = models[i].history.history['val_off_by_one_categorical_accuracy'][best_index]

        # Save:
        trainers[i].saveModel(
            'kfolds-{}-{}-fold{}-val_acc-{:.6f}'.format(
                model_type
                , label_type
                , i+1
                , max(models[i].history.history['val_acc'])
            )
        )

    # Print k-fold results:
    avg_results = np.average(results, axis=0)
    print("K-Fold results: avg_acc: {}, avg_off_by_1_acc: {}, avg_val_acc: {}, avg_val_off_by_1_acc: {}".format(
        avg_results[0], avg_results[1], avg_results[2], avg_results[3]
    ))
    print("Fold Details (acc, off_by_1_acc, val_acc, val_off_by_1_acc): ")
    print(results)

    # Generate k+1 confusion matrices (+1 for matrix across all folds):
    confusion_mtx(models, validation_generators, model_name=model_type, label_type=label_type, k=k)
    print()


# Can remove this later, only adding it so i can run this script from my IDE
def main_2():

    model_configs = [
        ("kfold_sameres_age", "age", "sameres"
            , "./saved_models/converge_sameres_age val_acc 0.564453125-Sat_28_Apr_2018_22_23_34/converge_sameres_age val_acc 0.564453125-Sat_28_Apr_2018_22_23_34.h5"
            , {
                "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose":True},
                "optimizer": "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 128, "stages": 3,
                "layers": 3, "dense": 2, "dense_size": 256, "img_size": 224, "grayscale": True
         })
        , ("kfold_sameres_gender", "gender", "sameres"
            , "./saved_models/converge_sameres_gender-val_acc-0.907906-Sun_29_Apr_2018_04_10_19/converge_sameres_gender-val_acc-0.907906-Sun_29_Apr_2018_04_10_19.h5"
            , {
                "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
                "optimizer_params": {"lr": 0.00015}, "batch_size": 64, "stages": 4, "layers": 2, "dense": 2,
                "dense_size": 256, "img_size": 224, "grayscale": True
           })
        , ("kfold_sameres_age_gender", "age_gender", "sameres"
            , "./saved_models/converge_sameres_age_gender-val_acc-0.552598-Sun_29_Apr_2018_13_13_33/converge_sameres_age_gender-val_acc-0.552598-Sun_29_Apr_2018_13_13_33.h5"
            , {
             "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
                "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 256, "stages": 3, "layers": 2,
             "dense": 1, "dense_size": 512, "img_size": 224, "grayscale": True
         })
        , ("kfold_resnet_age", "age", "resnet"
           , "./saved_models/converge_resnet_age-val_acc-0.556875-Sun_29_Apr_2018_06_46_40/converge_resnet_age-val_acc-0.556875-Sun_29_Apr_2018_06_46_40.h5"
           , {
               "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
                          "optimizer_params": {"lr": 0.0001}, "batch_size": 200, "stages": 3, "layers": 3,
               "dense": 2, "dense_size": 512, "img_size": 224, "grayscale": True
           })
        , ("kfold_resnet_gender", "gender", "resnet"
           , "./saved_models/converge_resnet_gender-val_acc-0.910389-Sun_29_Apr_2018_10_45_04/converge_resnet_gender-val_acc-0.910389-Sun_29_Apr_2018_10_45_04.h5"
           , {
               "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
               "optimizer_params": {"lr": 0.001}, "batch_size": 128, "stages": 2, "layers": 1, "dense": 3,
               "dense_size": 512, "img_size": 224, "grayscale": True
           })
        , ("kfold_resnet_age_gender", "age_gender", "resnet"
           , "./saved_models/converge_resnet_age_gender-val_acc-0.549659-Sun_29_Apr_2018_15_18_05/converge_resnet_age_gender-val_acc-0.549659-Sun_29_Apr_2018_15_18_05.h5"
           , {
               "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
                "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 64, "stages": 2, "layers": 3, "dense": 2,
               "dense_size": 128, "img_size": 224, "grayscale": True
           })
        , ("kfold_vgg16_age", "age", "vgg-16"
           , "./saved_models/converge_vgg16_age-val_acc-0.564628-Sun_29_Apr_2018_22_19_01/converge_vgg16_age-val_acc-0.564628-Sun_29_Apr_2018_22_19_01.h5"
           , {
               "reduce_lr_params": {"factor": 0.1, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
                "sgd", "optimizer_params": {"lr": 0.1, "decay": 0.005, "momentum": 0.85001, "nesterov": True},
               "batch_size": 80, "img_size": 224, "grayscale": True, "early_stop_patience": 7
           })


        # , ("kfold_vgg16_gender", "gender", "vgg-16"
        #    , "./saved_models//.h5"
        #    , {
        #        "reduce_lr_params": {"factor": 0.1, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
        #         "sgd", "optimizer_params": {"lr": 0.1, "decay": 0.005, "momentum": 0.95, "nesterov": True},
        #        "batch_size":
        #            80, "img_size": 224, "grayscale": True, "early_stop_patience": 7
        #    })
        # , ("kfold_vgg16_age_gender", "age_gender", "vgg-16"
        #    , "./saved_models//.h5"
        #    ,
        #    )
    ]

    dataset = "adience"
    grayscale = True
    img_size = 224
    epochs = 30
    K = 5

    for model_name, label_type, model_type, weights_path, trainer_config in model_configs:
        if "stages" in trainer_config:
            # (IF MODEL == RESNET || MODEL == ANDREYNET) && (not using pre-trained model):
            # Change values in this section for ResNet or AndreyNet architecture variable:
            architecture = {}
            architecture['stages'] = [[trainer_config["layers"], 64*(2**i)] for i in range(trainer_config["stages"])]
            architecture['dense'] = [trainer_config["dense_size"] for i in range(trainer_config["dense"])]
        print("Running k-folds cv for model_config: {}, label_type: {}, weights_path: '{}'".format(
            model_name
            , label_type
            , weights_path
        ))
        trainer_config["early_stop_patience"] = 6
        print("trainer_config: ", trainer_config)
        k_fold_cross_val(
            dataset,
            K,
            label_type,
            model_type,
            trainer_config,
            epochs=epochs,
            img_size=img_size,
            batch_size=trainer_config["batch_size"],
            grayscale=grayscale,
            architecture=architecture,
            model=weights_path
        )

if __name__ == '__main__':
    # main()
    main_2()
