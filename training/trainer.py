from time import strftime
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import keras
from keras.preprocessing.image import ImageDataGenerator
from twilio.rest import Client
from models.resnet import resnetBuilder



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
            optimizer = self.config["optimizer"]

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