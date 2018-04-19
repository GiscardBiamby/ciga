import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.layers import Add, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.engine.topology import Layer


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

	














