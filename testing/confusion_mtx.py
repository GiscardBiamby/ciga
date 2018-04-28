import sys
sys.path.append('../')
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import argparse
import models.vgg as cigaModels
import keras
import training.trainer as cigaTraining
from keras import optimizers
import matplotlib.pyplot as plt
from pylab import *

def plot_confusion(mtx, name):
    confusion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrices")
    if not os.path.exists(confusion_dir):
        os.makedirs(confusion_dir)
    path = confusion_dir + "/" + name
    plt.clf()
    fig = plt.figure()
    a_x = fig.add_subplot(111)
    res = a_x.imshow(mtx, cmap=cm.jet, interpolation='nearest')

    for i, j in ((x, y) for x in range(mtx.shape[0]) for y in range(mtx.shape[1])):
        a_x.annotate(str(mtx[i, j]), xy=(i, j))
    _ = fig.colorbar(res)
    plt.savefig(path)


def confusion_mtx(model, test_generator, model_name="VGG-16", k=None):
    if k:
        total = None
        for i in range(k):
            predictions = model[i].predict_generator(test_generator[i], use_multiprocessing=True)
            predictions = predictions.argmax(axis=-1)
            print(test_generator[i].classes)
            true_labels = np.array(test_generator[i].classes)
            confusion_mat = np.array(confusion_matrix(true_labels, predictions))
            if i == 0:
                total = confusion_mat
            else:
                total += confusion_mat
            plot_confusion(confusion_mat, model_name + "_confusion_mtx_fold_%d" % (i+1) + ".png")
        plot_confusion(total, model_name + "_confusion_mtx_all_%d_folds.png" % k)

    else:
        predictions = model.predict_generator(test_generator, use_multiprocessing=True)
        true_labels = np.array(test_generator.classes)
        confusion_mat = np.array(confusion_matrix(true_labels, predictions))
        plot_confusion(confusion_mat, model_name + "_confusion_mtx.png")
