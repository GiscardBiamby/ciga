import os
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout


def getModel_vgg11(image_shape, num_classes, name="vgg11"):
    ## VGG11 architecture: https://arxiv.org/pdf/1409.1556.pdf

    # This implementation is based on Configuration A from page 3 of 1409.1556.pdf, so 16 weight layers total:

    # Input (image):
    # Note, I read somewhere that for tensorflow the order matters for performance,
    # so check if this should be (1, img_size, img_size) instead?
    image_input = Input(shape=image_shape, name="image_input")

    # Note about pre-trained weights:
    # We have to do some potentially different pre-processing depending on which
    # pre-trained weights we use (if we use any). For example per-pixel mean-centering:
    # https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    # Group 1:
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g1_01")(image_input)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g1")(x)

    # Group 2:
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g2_01")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g2")(x)

    # Group 3:
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g3_01")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g3_02")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g3")(x)

    # Group 4:
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g4_01")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g4_02")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g4")(x)

    # Group 5:
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g5_01")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_g5_02")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g5")(x)

    # Final weight layers:
    # Note: The ReLu and Dropout stages here aren't part of original VGG paper, but the website by the authors
    # of the paper lists a slightly different version of the model that is their "best", which
    # includes these layers (http://www.robots.ox.ac.uk/~vgg/research/very_deep/):
    x = Flatten()(x)
    x = Dense(units=4096, name="final_fc_1")(x)
    x = Activation(activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=4096, name="final_fc_2")(x)
    x = Activation(activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=num_classes, activation='softmax', name='predictions')(x)
    #     x = Dense(units=1, activation="sigmoid", name="predictions")(x)

    # VGG paper says "Finally, to obtain a fixed-size vector of class scores for the image,
    # the class score map is spatially averaged (sum-pooled)."
    # Not sure if/where to add that in.
    # UDPATE: We probably don't need it. It's not mentioned in their
    # website when they define their "best" version of their vgg16 and vgg19 architectures.
    # x = GlobalAveragePooling2D()(x)

    model = Model(inputs=image_input, outputs=x, name=name)

    return model


def getModel_vgg16(image_shape, num_classes, name="vgg16", weights=""):
    ## VGG16 architecture: https://arxiv.org/pdf/1409.1556.pdf
    ## Also may be able to use VGG-Face pre-trained weights. Not sure if the VGG-Face
    ## architecture is the same as the VGG-16.
    ## Note: if the architectures don't match, they probably only differ by a small
    ## amount, so we can probably create a separate VGG-Face model based on our VGG16
    ## and then use the weights from Oxford: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

    # This implementation is based on Configuration D from page 3 of 1409.1556.pdf, so 16 weight layers total:

    # Input (image):
    # Note, I read somewhere that for tensorflow the order matters for performance,
    # so check if this should be (1, img_size, img_size) instead?
    image_input = Input(shape=image_shape, name="image_input")

    # Note about pre-trained weights:
    # We have to do some potentially different pre-processing depending on which
    # pre-trained weights we use (if we use any). For example per-pixel mean-centering:
    # https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    # Group 1:
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g1_01")(image_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g1_02")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g1")(x)

    # Group 2:
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g2_01")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g2_02")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g2")(x)

    # Group 3:
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g3_01")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g3_02")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g3_03")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g3")(x)

    # Group 4:
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g4_01")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g4_02")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g4_03")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g4")(x)

    # Group 5:
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g5_01")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g5_02")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1 ,1), activation="relu", name="conv2d_g5_03")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool_g5")(x)

    # Final weight layers:
    # Note: The ReLu and Dropout stages here aren't part of original VGG paper, but the website by the authors
    # of the paper lists a slightly different version of the paper model that is their "best", which
    # includes these layers (http://www.robots.ox.ac.uk/~vgg/research/very_deep/):
    x = Flatten()(x)
    x = Dense(units=4096, name="final_fc_1")(x)
    x = Activation(activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=4096, name="final_fc_2")(x)
    x = Activation(activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=num_classes, activation='softmax', name='predictions')(x)
    #     x = Dense(units=1, activation="sigmoid", name="predictions")(x)

    # VGG paper says "Finally, to obtain a fixed-size vector of class scores for the image,
    # the class score map is spatially averaged (sum-pooled)."
    # Not sure if/where to add that in.
    # UDPATE: We probably don't need it. It's not mentioned in their
    # website when they define their "best" version of their vgg16 and vgg19 architectures.
    # x = GlobalAveragePooling2D()(x)

    model = Model(inputs=image_input, outputs=x, name=name)

    if weights != "":
        weights_dir = "./weights/"
        weights_path = os.path.join(weights_dir, weights)
        model.load_weights(weights_path, by_name=True)

    return model


if __name__ == '__main__':
    model = getModel_vgg16((224, 224, 3), 2)