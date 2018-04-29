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

def plot_confusion(mtx, name, label_dct):
    confusion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrices")
    if not os.path.exists(confusion_dir):
        os.makedirs(confusion_dir)
    path = confusion_dir + "/" + name
    plt.clf()
    fig = plt.figure()
    a_x = fig.add_subplot(111)
    a_x.set_aspect(1)
    res = a_x.imshow(mtx, cmap=cm.RdBu, interpolation='nearest')
    plt.colorbar(res)
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            a_x.annotate(str(mtx[i, j]), xy=(j, i), horizontalalignment='center', verticalalignment='center')

    #_ = fig.colorbar(res)
    plt.xticks(list(label_dct.keys()), list(label_dct.values()), rotation=90)
    plt.yticks(list(label_dct.keys()), list(label_dct.values()))
    plt.savefig(path)


def confusion_mtx(model, test_generator, model_name="VGG-16", k=None):
    if k:
        total = None
        for i in range(k):
            predictions = model[i].predict_generator(test_generator[i], use_multiprocessing=True)
            label_dct = dict((v, k) for k, v in test_generator[i].class_indices.items())
            predictions = [label_dct[entry] for entry in predictions.argmax(axis=-1)]
            true_labels = [label_dct[entry] for entry in test_generator[i].classes]
            confusion_mat = np.array(confusion_matrix(true_labels, predictions, list(label_dct.values())))
            if i == 0:
                total = confusion_mat
            else:
                total += confusion_mat
            plot_confusion(confusion_mat, model_name + "_confusion_mtx_fold_%d" % (i+1) + ".png", label_dct)
        plot_confusion(total, model_name + "_confusion_mtx_all_%d_folds.png" % k, label_dct)

    else:
        predictions = model.predict_generator(test_generator, use_multiprocessing=True)
        label_dct = dict((v, k) for k, v in test_generator.class_indices.iteritems())
        predictions = [label_dct[entry] for entry in predictions.argmax(axis=-1)]
        true_labels = [label_dct[entry] for entry in test_generator.classes]
        confusion_mat = np.array(confusion_matrix(true_labels, predictions, list(label_dct.values())))
        plot_confusion(confusion_mat, model_name + "_confusion_mtx.png", label_dct)
