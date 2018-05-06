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

def plot_confusion(mtx, save_path, label_dct):

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
    plt.savefig(save_path)


def confusion_mtx2(model, test_generator, saved_model_dir, fold_num):
    predictions = model.predict_generator(test_generator, use_multiprocessing=True)
    label_dct = dict((v, k) for k, v in test_generator.class_indices.items())
    # print("Predictions: ", type(predictions), predictions)
    # print("len(pred): ", len(predictions))
    if type(predictions) == type(list()):
        predictions = np.array(predictions)

    predictions = [label_dct[entry] for entry in predictions.argmax(axis=-1)]
    true_labels = [label_dct[entry] for entry in test_generator.classes]
    confusion_mat = np.array(confusion_matrix(true_labels, predictions, list(label_dct.values())))

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    confusion_data_path = os.path.join(saved_model_dir, "confusion_mtx_fold_%d" % (fold_num + 1) + "_numpy_data.txt")
    np.savetxt(confusion_data_path, confusion_mat, fmt='%d')
    plot_path = os.path.join(saved_model_dir, "confusion_mtx_fold_%d" % (fold_num + 1) + ".png")
    plot_confusion(confusion_mat, plot_path, label_dct)
    return confusion_mat