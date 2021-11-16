import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import seaborn as sn

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics


def save_cr_and_cm(label_list, list_of_refer, list_of_pred, cr_save_path="/home/tmp/pycharm_project_61/output/wordpiece/classification_report.csv",
                   cm_save_path="output/test_confusion_matrix.png"):
    """ print classification report and confusion matrix """

    target_names = []
    bio_tag_list = list(label_list.values())
    for label_name in bio_tag_list:
        if label_name in ['[PAD]']:
            continue
        else:
            target_names.append(label_name)

    label_index_to_print = list(range(1, len(label_list)))

    # --- get f1-score report by tag(class) --- #
    temp_cr = classification_report(y_true=list_of_refer, y_pred=list_of_pred)
    print(temp_cr)
    # print(classification_report(y_true=list_of_refer, y_pred=list_of_pred, target_names=target_names))
    cr_dict = classification_report(y_true=list_of_refer, y_pred=list_of_pred, output_dict=True)
    df = pd.DataFrame(cr_dict).transpose()
    df.to_csv(cr_save_path)
    print('\nSave [f1-score report by tag] at {}\n'.format(cr_save_path))
    # --- get f1-score report by tag(class) --- #

    # --- get confusion matrix --- #
    #np.set_printoptions(precision=2)

    #plot_confusion_matrix(y_true=list_of_refer, y_pred=list_of_pred, classes=target_names, labels=label_index_to_print, normalize=False)
    #plt.savefig(cm_save_path)
    #print('\nSave Confusion Matrix at {}\n'.format(cm_save_path))
    # --- get confusion matrix --- #

    return True

import itertools


def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # --- plot 크기 조절 --- #
    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['figure.figsize'] = [50, 50]  # plot 크기
    plt.rcParams.update({'font.size': 10})
    # --- plot 크기 조절 --- #

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # --- bar 크기 조절 --- #
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar 크기 조절 --- #
    # ax.figure.colorbar(im, ax=ax)

    sorted_classes = sorted(classes)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           # xticklabels=sorted_classes, yticklabels=sorted_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax