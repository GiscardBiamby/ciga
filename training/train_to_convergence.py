import sys
sys.path.append('../')

from models.AndreyNet import AndreyNetBuilder
from models.vgg import getModel_vgg16, getModel_vgg11
from models.resnet import resnetBuilder

import training.trainer as cigaTraining
from keras import optimizers
import numpy as np
import gc
import keras

ENABLE_SMS = False
ENABLE_CHECKPOINTS = False
TEST_MODE = False

def train(
        dataset_path
        , architecture
        , trainer_config
        , model_key
        , epochs=1
        , img_size=50
        , batch_size=50
        , grayscale=True
):
    # Build data generator:
    train_generator, validation_generator = cigaTraining.generatorsBuilder(
        dataset_path, img_dims=(img_size, img_size), batch_size=batch_size, grayscale=grayscale
    )

    # Build Model:
    if "sameres" in model_key:
        model = AndreyNetBuilder(architecture, train_generator.image_shape, train_generator.num_classes)
    elif "resnet" in model_key:
        model = resnetBuilder(architecture, train_generator.image_shape, train_generator.num_classes)
    elif "vgg" in model_key:
        model = getModel_vgg16(train_generator.image_shape, train_generator.num_classes)
    else:
        raise NotImplementedError("Unknown architecture: ", model_key)

    # Train:
    trainer = cigaTraining.BasicTrainer(
        model=model
        , config=trainer_config
        , enable_sms=ENABLE_SMS
        , enable_checkpoints=ENABLE_CHECKPOINTS
    )

    try:
        model = trainer.train(
            validation_generator
            , train_generator
            , batch_size=batch_size
            , epochs=epochs
        )
        trainer.saveModel(model_key + "-val_acc-{0:.6f}".format(max(model.history.history['val_acc'])))
    except Exception as e:
        print("ERROR while training model w/ config: ", trainer_config)
        print("Error: ", e)
        # Save to prevent tryign to re-run same config again:
        trainer.saveModel(model_key + 'ERROR', saveConfigOnly=True)

    # Try to deal w/ GPU OOM errors:
    del model
    del trainer
    gc.collect()
    cigaTraining.BasicTrainer.clearGpuSession()


def trainToConvergence():
    if TEST_MODE:
        dataset_path_base = '../datasets/processed_small/wiki/'
        epochs = 2
    else:
        dataset_path_base = '../datasets/processed/imdbwiki/'
        epochs = 75
    # List of tuples describing which models to train. Tuple items:
    #   tuple[0]: model_key, AKA some unique string description of the model
    #   tuple[1]: label type. One of the following: "age", "gender", "age_gender"
    #   tuple[2]: trainer_config, a dictionary of hyper params, and other settings used to create and train model.
    models = [
        # ("converge2_sameres_age", "age", {"reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose":
        #     True},
        #               "optimizer": "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 128, "stages": 3,
        #                  "layers": 3, "dense": 2, "dense_size": 256, "img_size": 224, "grayscale": True})
        # ,
        # ("converge2_sameres_gender", "gender"
        #    , {
        #        "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
        #        "optimizer_params": {"lr": 0.00015}, "batch_size": 64, "stages": 4, "layers": 2, "dense": 2,
        #                       "dense_size": 256, "img_size": 224, "grayscale": True
        #    })
        # , ("converge2_sameres_age_gender", "age_gender"
        #    , {
        #        "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
        #         "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 256, "stages": 3, "layers": 2,
        #                           "dense": 1, "dense_size": 512, "img_size": 224, "grayscale": True
        #    })
        #
        # , ("converge2_resnet_age", "age"
        #    , {
        #        "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
        #                   "optimizer_params": {"lr": 0.0001}, "batch_size": 200, "stages": 3, "layers": 3,
        #        "dense": 2, "dense_size": 512, "img_size": 224, "grayscale": True
        #    })
        # ("converge2_resnet_gender", "gender"
        #    , {
        #        "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08}, "optimizer": "adam",
        #        "optimizer_params": {"lr": 0.001}, "batch_size": 64, "stages": 2, "layers": 1, "dense": 3,
        #        "dense_size": 512, "img_size": 224, "grayscale": True
        #    })
        # ("converge2_resnet_age_gender", "age_gender"
        #    , {
        #        "reduce_lr_params": {"factor": 0.2, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
        #        "adam", "optimizer_params": {"lr": 0.001}, "batch_size": 64, "stages": 2, "layers": 3, "dense": 2,
        #         "dense_size": 128, "img_size": 224, "grayscale": True
        #   })
        #
        # , ("converge2_vgg16_age", "age"
        #    , {
        #         "reduce_lr_params": {"factor": 0.1, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
        #         "sgd", "optimizer_params": {"lr": 0.1, "decay": 0.005, "momentum": 0.85001, "nesterov": True},
        #         "batch_size": 80, "img_size": 224, "grayscale": True, "early_stop_patience": 9
        #    })
        ("converge2_vgg16_gender", "gender"
           , {
               "reduce_lr_params": {"factor": 0.1, "patience": 3, "min_lr": 1e-08, "verbose": True}, "optimizer":
                "sgd", "optimizer_params": {"lr": 0.100000001, "decay": 0.005, "momentum": 0.95, "nesterov": True},
             "batch_size":
                80, "img_size": 224, "grayscale": True, "early_stop_patience": 9
           })
        # , ("vgg16_age_gender", "age_gender"
        #    ,
        #    )
    ]

    i = 1
    total = len(models)

    for model_key, split, trainer_config in models:
        dataset_path = dataset_path_base + split + "/"
        print('Traning model %d/%d (%s) to convergence...' % (i, total, model_key))
        print('+====================================================================+')

        architecture = {}
        if "stages" in trainer_config:
            architecture['stages'] = [[trainer_config["layers"], 64*(2**i)] for i in range(trainer_config["stages"])]
            architecture['dense'] = [trainer_config["dense_size"] for i in range(trainer_config["dense"])]

        trainer_config["early_stop_patience"] = 9

        if cigaTraining.configHasAlreadyRun(trainer_config, model_key):
            print("Skipping training for previously run config: ", trainer_config)
            i += 1
            continue

        print("Starting training w/ config: ", trainer_config)
        train(
            dataset_path
            , architecture
            , trainer_config
            , model_key
            , epochs=epochs
            , img_size=trainer_config["img_size"]
            , batch_size=trainer_config["batch_size"]
            , grayscale=trainer_config["grayscale"]
        )

        print('+====================================================================+')
        i += 1
    print('DONE.')





if __name__ == '__main__':
    cigaTraining.BasicTrainer.clearGpuSession()
    trainToConvergence()
    # BasicTrainer.gatherHyperParamResults("AndreyNet val_acc", "./andreynet_hyperparam_results.csv")
    # started: 11:50pm










