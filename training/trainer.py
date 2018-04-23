from time import strftime
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import keras
from keras.preprocessing.image import ImageDataGenerator
from twilio.rest import Client
from models.resnet import resnetBuilder
import os
import json
from keras.models import model_from_json, load_model
import matplotlib.pyplot as plt
from keras import optimizers
import numpy

# Maybe this should go into it's own file and/or class
def generatorsBuilder(dataset_path, img_dims=(50, 50), batch_size=32, grayscale=True):
    train_data_dir = dataset_path + 'train/'
    valid_data_dir = dataset_path + 'valid/'

    color_mode = 'grayscale' if grayscale else 'rgb'

    train_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        color_mode=color_mode,
                                                        target_size=img_dims,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    valid_datagen = ImageDataGenerator(samplewise_center=True,
                                       samplewise_std_normalization=True)

    valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
                                                        color_mode=color_mode,
                                                        target_size=img_dims,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    return train_generator, valid_generator


class BasicTrainer(object):

    def __init__(self, model, config, enable_checkpoints=False, enable_sms=False):
        self.model = model
        self.config = config
        self.enable_checkpoints = enable_checkpoints
        self.enable_sms = enable_sms
        self.callbacks = []
        self.init_callbacks()


    def init_callbacks(self):
        if (self.enable_checkpoints):
            self.callbacks.append(
                ModelCheckpoint(
                    filepath='../weights/{}-{}.hdf5'.format(self.model.name, strftime("%a_%d_%b_%Y_%H_%M_%S"))
                    , verbose=1
                    , save_best_only=True
                )
            )

        # I think we always use reduceLR, but if we ever don't, just add an if statement
        # to only append this callback if the reduce_lr_params key exists in self.config:
        self.callbacks.append(ReduceLROnPlateau(**(self.config["reduce_lr_params"])))
        self.callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'))

        # Add if statement, or just make the caller use self.addCallback, since not everyoe will use this:
        if self.enable_sms:
            class DoneAlert(keras.callbacks.Callback):
                def on_train_end(self, logs={}):
                    account_sid = "ACdb60c905cd24f1bd71e6b49efb7a75c4"
                    auth_token = "607eb22c4134ad6904ce2ad87d066e58"
                    to = "+19258587735"
                    from_ = "+16504828933"
                    client = Client(account_sid, auth_token)

                    max_val_acc = max(self.model.history.history['val_acc'])
                    min_val_loss = min(self.model.history.history['val_loss'])
                    msg = "Training Ended. val_acc=" + str(max_val_acc) + ' \n min val_loss=' + str(min_val_loss)
                    message = client.messages.create(to=to, from_=from_, body=msg)

            self.callbacks.append(DoneAlert())

        # self.callbacks.append(TensorBoard(log_dir="./logs/{}".format(strftime("%a, %d %b %Y %H:%M:%S"))))



    def addCallback(self, callback):
        self.callbacks.append(callback)


    def train(self
        , validation_generator
        , train_generator
        , batch_size=32
        , epochs=10
      ):
        num_valid = validation_generator.samples
        num_train = train_generator.samples

        # Default to Adam if config doesn't specify an optimizer:

        optimizer = "adam"
        if ("optimizer" in self.config and self.config["optimizer"] is not None):
            if self.config["optimizer"] == "adam":
                optimizer = optimizers.Adam(**(self.config["optimizer_params"]))
            elif self.config["optimizer"] == "sgd":
                optimizer = optimizers.SGD(**(self.config["optimizer_params"]))
            else:
                raise NotImplementedError(self.config["optimizer"])

        self.model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        history = self.model.fit_generator(train_generator,
                                      validation_data=validation_generator,
                                      validation_steps=num_valid//batch_size,
                                      steps_per_epoch=num_train//batch_size,
                                      epochs=epochs,
                                      callbacks=self.callbacks,
                                      verbose=1,
                                      use_multiprocessing=True,
                                      workers=4)
        return self.model


    def saveModel(self, name):
        """
        This function should save as much info as possible about the model, as well as the model + weights.
        Make some kind of json file or similar, alongside the .h5 file. json should contain all the config info about
        how the model was trained (hyperparams, etc). Plots, training accuracy, history, etc.
        So that once we train, we have access to all the info so we can write up results later.

        :param name:
        :return:
        """
        name = "{}-{}".format(name, strftime("%a_%d_%b_%Y_%H_%M_%S"))
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        savedModelDir = os.path.join(project_root, "models/saved_models/{}/".format(name))
        print("savedModelsPath: ", savedModelDir)
        if not os.path.exists(savedModelDir):
            os.makedirs(savedModelDir)

        # Save model and weights:
        self.model.save(os.path.join(savedModelDir, name + ".h5"))
        with open(os.path.join(savedModelDir, 'model_architecture.json'), 'w') as f:
            f.write(self.model.to_json())

        # Save summary:
        with open(os.path.join(savedModelDir, 'model_summary.txt'), 'w') as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + '\n'))

        # Save model diagram:
        keras.utils.plot_model(self.model
                               , to_file=os.path.join(savedModelDir, 'model.png')
                               , show_shapes=True
                               , show_layer_names=True
                               , rankdir='TB'
                               )
        # Save Plots:
        self.saveTrainingPlots(savedModelDir)

        # Save trainer config:
        with open(os.path.join(savedModelDir, 'trainer_config.json'), 'w') as fp:
            json.dump(self.config, fp, cls=CustJsonEncoder)

    def saveTrainingPlots(self, savedModelDir):
        """
        Saves plots of loss/val_loss, acc/val_acc, and learning_rate to savedModelDir
        :param savedModelDir:
        :return:
        """
        val_loss, val_acc, loss, acc, lr = self.model.history.history.values()
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.savefig(os.path.join(savedModelDir, "loss.png"))
        plt.close()

        plt.plot(acc, label="acc")
        plt.plot(val_acc, label="val_acc")
        plt.legend()
        plt.savefig(os.path.join(savedModelDir, "accuracy.png"))
        plt.close()

        plt.plot(lr, label="lr")
        plt.legend()
        plt.savefig(os.path.join(savedModelDir, "learning_rate.png"))
        plt.close()



class CustJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(CustJsonEncoder, self).default(obj)