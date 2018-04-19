import sys
sys.path.append('../')

import models.vgg as cigaModels
import training.trainer as cigaTraining
from keras import optimizers



def trainVgg16(dataset_path, trainer_config, epochs=1, img_size=50, batch_size=32, grayscale=True):

    num_channels = 1 if grayscale else 3
    img_shape = (img_size, img_size, num_channels)

    # Build data generator:
    train_generator, validation_generator = cigaTraining.generatorsBuilder(
        dataset_path, img_dims=(img_size, img_size), batch_size=batch_size, grayscale=grayscale
    )
    num_classes = train_generator.num_classes

    # Build Model:
    model = cigaModels.getModel_vgg16(img_shape, num_classes)

    # Train:
    trainer = cigaTraining.BasicTrainer(
        model = model
        , config = trainer_config
        , enable_sms = False
    )
    history, model = trainer.train(
        validation_generator
        , train_generator
        , batch_size = batch_size
        , epochs = epochs
    )



    # Do other stuff, generate plots, save results, etc.

if __name__ == '__main__':
    dataset_name = "gender"
    dataset_path = "../datasets/processed/wiki/{}/".format(dataset_name)
    grayscale = True
    img_size = 224
    batch_size = 64
    epochs = 1
    trainer_config = {
            "reduce_lr_params": {
                "factor":  0.2
                , "patience": 1
                , "min_lr": 1e-8
                , "verbose": 1
            }
            , "optimizer": optimizers.SGD(lr=1e-2, decay=5e-4, momentum=0.9, nesterov=True)
        }
    trainVgg16(dataset_path, trainer_config, epochs=epochs, img_size=img_size, batch_size=batch_size, grayscale=grayscale)