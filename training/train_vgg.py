import sys
sys.path.append('../')

import models.vgg as cigaModels
import training.trainer as cigaTraining
from keras import optimizers
import numpy as np
import gc
import keras

model_key = "vgg16_age val_acc "

## TODO: Put all these params into trainer_config, so they get saved into trainer.config.json (see BasicTrainer.saveModel())
def trainVgg16(dataset_path, trainer_config, epochs=1, img_size=50, batch_size=32, grayscale=True):

    num_channels = 1 if grayscale else 3
    img_shape = (img_size, img_size, num_channels)

    # Build data generator:
    train_generator, validation_generator = cigaTraining.generatorsBuilder(
        dataset_path, img_dims=(img_size, img_size), batch_size=batch_size, grayscale=grayscale
    )

    # Build Model:
    model = cigaModels.getModel_vgg16(img_shape, train_generator.num_classes)

    # Train:
    trainer = cigaTraining.BasicTrainer(
        model = model
        , config = trainer_config
        , enable_sms = False
    )
    try:
        model = trainer.train(
            validation_generator
            , train_generator
            , batch_size = batch_size
            , epochs = epochs
        )
        trainer.saveModel(model_key + "{0:.6f}".format(max(model.history.history['val_acc'])) )
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


def Vgg16RandomSearch(numChoices=2):
    dataset_path = '../datasets/processed/wiki/age/'

    epochs = 35

    img_size = 224
    grayscale = True

    batch_sizes = [80]
    optimizers = ["sgd"]
    learning_rates = [1e-1]
    decay_rates = [5e-3]
    momentums = [.975, .95001, .925, .90001, .875, .85001]

    print("batch_sizes = {}".format(batch_sizes))
    print("optimizers = {}".format(optimizers))
    print("learning_rates = {}".format(learning_rates))
    print("decay_rates = {}".format(decay_rates))
    print("momentums = {}".format(momentums))

    i = 1
    total = 64

    for batch_size in batch_sizes:
        for optimizer in optimizers:
            for decay in decay_rates:
                for momentum in momentums:
                    for lr in learning_rates:
                        print('Running %d/%d' % (i, total))
                        print('+====================================================================+')

                        if optimizer == "adam":
                            optimizer_params = {"lr": lr}
                        else:
                            optimizer_params = {
                                "lr": lr, "decay": decay, "momentum": momentum, "nesterov": True
                            }

                        trainer_config = {
                            "reduce_lr_params": {
                                "factor": 0.1
                                , "patience": 3
                                , "min_lr": 1e-8
                                , "verbose": 1
                            }
                            , "optimizer": optimizer
                            , "optimizer_params": optimizer_params
                            , "batch_size": batch_size
                            , "img_size": img_size
                            , "grayscale": grayscale
                            # Make this X times as big as reduce_lr_params to give it a chance to learn after reducing
                            # learning_rate. X = # times you want lr to reduce:
                            , "early_stop_patience": 3 * 3
                        }


                        if cigaTraining.configHasAlreadyRun(trainer_config, model_key.strip()):
                            print("Skipping training for previously run config: ", trainer_config)
                            i += 1
                            continue

                        print("Starting training w/ config: ", trainer_config)

                        trainVgg16(dataset_path, trainer_config, epochs=epochs, img_size=img_size,
                                   batch_size=batch_size, grayscale=grayscale)
                        print('+====================================================================+')
                        i += 1
    print('DONE.')

if __name__ == '__main__':
    cigaTraining.BasicTrainer.clearGpuSession()
    Vgg16RandomSearch()
    # cigaTraining.BasicTrainer.gatherHyperParamResults("vgg16_gender", "./vgg16_hyperparam_results_gender_wiki.csv")

