import sys
sys.path.append('../')

import models.resnet as cigaModels
import training.trainer as cigaTraining
from keras import optimizers


def trainResnet(dataset_path, architecture, trainer_config, epochs=1, img_size=(50, 50), batch_size=50, grayscale=True):

    # Build data generator:
    train_generator, validation_generator = cigaTraining.generatorsBuilder(dataset_path, img_dims=img_size, batch_size=batch_size, grayscale=grayscale)
    
    # Build Model:
    model = cigaModels.resnetBuilder(architecture, train_generator.image_shape, train_generator.num_classes)

    # Train:
    trainer = cigaTraining.BasicTrainer(model = model, config = trainer_config, enable_sms = False)

    model = trainer.train(
        validation_generator
        , train_generator
        , batch_size = batch_size
        , epochs = epochs
    )


if __name__ == '__main__':
    dataset_path = '../datasets/processed/wiki/age/'
    img_size = (50, 50)
    batch_size = 32
    grayscale = True
    epochs = 1

    architecture = {'stages': [[1, 1]],
                 'dense': [32]}

    optimizer = optimizers.Adam()

    trainer_config = {
            "reduce_lr_params": {
                "factor":  0.2
                , "patience": 1
                , "min_lr": 1e-8
                },
                "optimizer": optimizer
            }
    trainResnet(dataset_path, architecture, trainer_config, epochs=epochs, img_size=img_size, batch_size=batch_size, grayscale=grayscale)














