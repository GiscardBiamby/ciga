import sys
sys.path.append('../')

import models.AndreyNet as cigaModels
import training.trainer as cigaTraining
from keras import optimizers
import numpy as np

## TODO: Put all these params into trainer_config, so they get saved into trainer.config.json (see BasicTrainer.saveModel())
def trainAndreyNet(dataset_path, architecture, trainer_config, epochs=1, img_size=50, batch_size=50, grayscale=True):

    # Build data generator:
    train_generator, validation_generator = cigaTraining.generatorsBuilder(dataset_path, img_dims=(img_size, img_size) , batch_size=batch_size, grayscale=grayscale)
    
    # Build Model:
    model = cigaModels.AndreyNetBuilder(architecture, train_generator.image_shape, train_generator.num_classes)

    # Train:
    trainer = cigaTraining.BasicTrainer(model = model, config = trainer_config, enable_sms = False)

    model = trainer.train(
        validation_generator
        , train_generator
        , batch_size = batch_size
        , epochs = epochs
    )
    trainer.saveModel('AndreyNet val_acc ' + str(max(model.history.history['val_acc'])))

def AndreyNetRandomSearch(numChoices=2):
    dataset_path = '../datasets/processed/wiki/age/'

    epochs = 20
    
    img_size = 224
    grayscale = True
    optimizer = 'adam'

    batch_sizes = [64, 128, 256, 512, 1024]
    num_stages = [2, 3, 4, 5]
    num_layers = [2, 3, 4, 5]
    num_denses = [1, 2, 3]
    dense_sizes = [256, 512, 1024]
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]

    batch_sizes = np.random.choice(batch_sizes, min(numChoices, len(batch_sizes)), replace=False)
    num_stages = np.random.choice(num_stages, min(numChoices, len(num_stages)), replace=False)
    num_layers = np.random.choice(num_layers, min(numChoices, len(num_layers)), replace=False)
    num_denses = np.random.choice(num_denses, min(numChoices, len(num_denses)), replace=False)
    dense_sizes = np.random.choice(dense_sizes, min(numChoices, len(dense_sizes)), replace=False)
    learning_rates = np.random.choice(learning_rates, min(numChoices, len(learning_rates)), replace=False)

    i = 1
    total = numChoices**6

    for batch_size in batch_sizes:
        for stages in num_stages:
            for layers in num_layers:
                for dense in num_denses:
                    for dense_size in dense_sizes:
                        for lr in learning_rates:
                            print('Running %d/%d' % (i, total))
                            print('+====================================================================+')
                            architecture = {}
                            architecture['stages'] = [[layers, 64*(2**i)] for i in range(stages)]
                            architecture['dense'] = [dense_size for i in range(dense)]

                            trainer_config = {
                                    "reduce_lr_params": {
                                        "factor":  0.2
                                        , "patience": 3
                                        , "min_lr": 1e-8
                                        }
                                        , "optimizer": optimizer
                                        , "optimizer_params": {"lr" : lr}
                                        , "batch_size" : batch_size
                                        , "stages" : stages
                                        , "layers" : layers
                                        , "dense" : dense
                                        , "dense_size" : dense_size
                                        , "img_size" : img_size
                                        , "grayscale" : grayscale
                                    }

                            if cigaTraining.configHasAlreadyRun(trainer_config, "ResNet val_acc"):
                                print("Skipping training for previously run config: ", trainer_config)
                                continue

                            print("Starting training w/ config: ", trainer_config)
                            trainAndreyNet(dataset_path, architecture, trainer_config,
                                        epochs=epochs, img_size=img_size, batch_size=batch_size,
                                        grayscale=grayscale)
                            print('+====================================================================+')  
                            i += 1  
    print('DONE.')





if __name__ == '__main__':
    AndreyNetRandomSearch()











