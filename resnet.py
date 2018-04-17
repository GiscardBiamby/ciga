from time import strftime
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Add, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import pydot
import graphviz
from keras.utils import plot_model
from keras.engine.topology import Layer
from twilio.rest import Client

class DoneAlert(keras.callbacks.Callback):
    def on_train_end(self, logs={}):
        account_sid = "ACdb60c905cd24f1bd71e6b49efb7a75c4"
        auth_token = "607eb22c4134ad6904ce2ad87d066e58"
        to = "+19258587735"
#         to = "+15102900156"
        from_ = "+16504828933"
        client = Client(account_sid, auth_token)
        
        max_val_acc = max(self.model.history.history['val_acc'])
        min_val_loss = min(model.history.history['val_loss'])
        msg = "Training Ended. val_acc="+str(max_val_acc)+' \n min val_loss=' + str(min_val_loss)
        message = client.messages.create(to=to, from_=from_, body=msg)

def generatorsBuilder(dataset_path, img_size=(50, 50), batch_size=32, grayscale=True):
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
													    target_size=img_size,
													    batch_size=batch_size,
													    class_mode='categorical')

	valid_datagen = ImageDataGenerator(samplewise_center=True, 
										samplewise_std_normalization=True)

	valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
														color_mode=color_mode,
														target_size=img_size,
														batch_size=batch_size,
														class_mode='categorical')
	return train_generator, valid_generator

def resnetBuilder(architecture, img_shape, num_classes):
	first_filters = architecture['stages'][0][1]
	inputs = Input(shape=img_shape)

	x = BatchNormalization()(inputs)
	x = Activation('relu')(x)
	x = Conv2D(filters=first_filters, kernel_size=7, strides=(2, 2), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=first_filters, kernel_size=7,  padding='same')(x)

	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	res = x

	# architecture = {'stages': [[1, 4], [1, 8]],
	# 				 'dense': [256]}

	for stage in architecture['stages']:
	    depth ,filters = stage
	    
	    # halves residual spatial dimentions to match main branch shape
	    if stage != 0:
	        res = Conv2D(filters=filters, kernel_size=(1, 1), strides=2, padding='same')(res)
	        
	    for i in range(depth):        
	        # First resnet layer
	        x = BatchNormalization()(x)
	        x = Activation('relu')(x)
	        # halve the spatial dimentions every time number of filters doubles, but not in the first stage
	        if i == 0 and stage != 0: 
	            x = Conv2D(filters=filters, kernel_size=3, padding='same', strides=(2, 2))(x)
	        else:
	            x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
	        x = BatchNormalization()(x)
	        x = Activation('relu')(x)
	        x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
	        
	        # Second resnet layer
	        x = BatchNormalization()(x)
	        x = Activation('relu')(x)
	        x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
	        x = BatchNormalization()(x)
	        x = Activation('relu')(x)
	        x = Conv2D(filters=filters, kernel_size=3, padding='same')(x)
	        x = Add()([x, res])
	        res = x
	    
	x = AveragePooling2D(pool_size=(2, 2))(x)

	x = Flatten()(x)
	x = Dropout(0.5)(x)

	for size in architecture['dense']:
	    x = Dense(size, activation='relu')(x)
	    
	predictions = Dense(num_classes, activation='softmax')(x)

	model = Model(inputs=inputs, outputs=predictions)
	return model

def train(dataset_path, architecture, img_size=(50, 50), batch_size=32, grayscale=True, epochs=10,
		lr_factor=0.2, lr_patience=3, min_lr=1e-7, callbacks=None):
	train_generator, validation_generator = generatorsBuilder(path, img_size=img_size, batch_size=batch_size, grayscale=grayscale)
	num_valid = validation_generator.samples
	num_train = train_generator.samples

	model = resnetBuilder(architecture, train_generator.image_shape, train_generator.num_classes)

	model.compile(optimizer='adam',
				loss='categorical_crossentropy',
            	metrics=['accuracy'])

	history = model.fit_generator(train_generator,
	                              validation_data=validation_generator,
	                              validation_steps=num_valid//batch_size,
	                              steps_per_epoch=num_train//batch_size, 
	                              epochs=epochs,
	                              callbacks=callbacks,
	                              verbose=1,
	                              use_multiprocessing=True,
	                              workers=4)
	return history, model


if __name__ == '__main__':

	path = './datasets/processed/wiki/age/'
	img_size = (50, 50)

	architecture = {'stages': [[1, 1]],
					 'dense': [32]}

	batch_size = 32
	grayscale = True
	epochs = 1
	lr_factor = 0.2
	lr_patience = 1
	min_lr = 1e-8

	checkpointer = ModelCheckpoint(filepath='./weights/{}.hdf5'.format(strftime("%a_%d_%b_%Y_%H_%M_%S")), verbose=1, save_best_only=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=min_lr, verbose=1)
	sms_alert = DoneAlert()
	# # tensorboard = TensorBoard(log_dir="./logs/{}".format(strftime("%a, %d %b %Y %H:%M:%S")))
	# # callbacks = [reduce_lr, checkpointer, sms_alert]
	# # callbacks = [reduce_lr, checkpointer]
	callbacks = [reduce_lr]

	history, model = train(path, architecture, img_size=img_size, batch_size=batch_size, epochs=1, lr_patience=1, min_lr=1e-10, callbacks=callbacks)


	














