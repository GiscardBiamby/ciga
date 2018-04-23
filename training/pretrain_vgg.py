"""
Pre-train vgg16 model on IMDB + WIKI and then test prediction on adience:
This is just a test script to verify that we can pretrain, save the weight
"""
import sys
sys.path.append('../')
import models.vgg as cigaModels
import training.trainer as cigaTraining
from keras import optimizers
from keras.models import load_model
import os
import glob


## TODO: Put all these params into trainer_config, so they get saved into trainer.config.json (see BasicTrainer.saveModel())
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
    model = trainer.train(
        validation_generator
        , train_generator
        , batch_size = batch_size
        , epochs = epochs
    )
    trainer.saveModel("wiki_gender_pretrained_vgg16")

def wikiPretraining():
    dataset_name = "gender"
    dataset_path = "../datasets/processed/wiki/{}/".format(dataset_name)
    grayscale = False
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
            , "optimizer": "sgd"
            , "optimizer_params": { "lr": 1e-2, "decay": 5e-4, "momentum": 0.9, "nesterov": True }
        }
    trainVgg16(dataset_path, trainer_config, epochs=epochs, img_size=img_size, batch_size=batch_size, grayscale=grayscale)

def get_saved_model_path(saved_model_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_models_dir = os.path.join(project_root, "models/saved_models/")
    print("saved_models_dir: ", saved_models_dir)
    matches = list(glob.glob("{}/{}*/".format(saved_models_dir, saved_model_name)))

    if len(matches)>1:
        raise Exception(
            "ERROR! Found more than one saved model matching saved_model_name='{}'".format(
                saved_model_name
            )
        )
    if len(matches==0):
        raise Exception(
            "ERROR! No matching saved models found for saved_model_name='{}'".format(
                saved_model_name
            )
        )

    saved_model_path = glob.glob("{}/{}*.h5".format(matches[0], saved_model_name))
    return saved_model_path[0]

# Note: it will be annoying to lookup the file name manually, since it has the timestamp as part of the file name.
# we should probably just say the saved model names must be unique and remove the timestamp from the saving process.
def predictAdience(saved_model_name):
    # Load model:
    img_size = 224
    grayscale = False
    num_channels = 1 if grayscale else 3
    img_shape = (img_size, img_size, num_channels)
    num_classes = 8

    file_path = get_saved_model_path(saved_model_name)
    model = load_model(file_path)

if __name__ == '__main__':
    # wikiPretraining()
    # predictAdience()
    print(get_saved_model_path("foo"))