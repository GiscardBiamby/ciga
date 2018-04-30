import sys
import os
import numpy as np
import argparse
import models.vgg as vgg
import models.resnet as resnet
import models.AndreyNet as andreynet
import training.trainer as cigaTraining
import tensorflow as tf
import keras
from keras import optimizers
from keras import backend as K
from testing.confusion_mtx import *

def main():
    print("1")
    model = keras.models.load_model("./saved_models/converge_sameres_age val_acc "
                            "0.564453125-Sat_28_Apr_2018_22_23_34/converge_sameres_age val_acc "
                            "0.564453125-Sat_28_Apr_2018_22_23_34.h5",
                            custom_objects={'off_by_one_categorical_accuracy':
                                                cigaTraining.off_by_one_categorical_accuracy})
    print("2")
    print(model.__dict__)
    print("3")
    # print(model.history.history["off_by_one_categorical_accuracy"])
    print("4")



main()
