# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:11:02 2018

@author: HP
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt





x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_test  = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[10], dtype=np.str)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 2), random_state=1)

clf.n_outputs_ = 2
 
clf.fit(x_train, y_train)
 
 
pdd = clf.predict(x_test)
 
M = confusion_matrix(y_test, pdd)

print(M)
#
#Y_T = label_binarize(y_test, classes=['NEG', 'POS'])
#Y_P = label_binarize(pdd, classes=['NEG', 'POS'])
#
#n_classes = Y_T.shape[1]
#
#average_precision = average_precision_score(Y_T, Y_P)
#precision, recall, _ = precision_recall_curve(Y_T, Y_P)
#
#
#plt.step(recall, precision, color='b', alpha=0.2,
#         where='post')
#plt.fill_between(recall, precision, step='post', alpha=0.2,
#                 color='b')
#
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
#
##########################
#
#
## For each class
#precision = dict()
#recall = dict()
#average_precision = dict()
#for i in range(n_classes):
#    precision[i], recall[i], _ = precision_recall_curve(Y_T[:, i],
#                                                        Y_P[:, i])
#    average_precision[i] = average_precision_score(Y_T[:, i], Y_P[:, i])
#    
## A "micro-average": quantifying score on all classes jointly
#precision["micro"], recall["micro"], _ = precision_recall_curve(Y_T.ravel(),
#    Y_P.ravel())
#average_precision["micro"] = average_precision_score(Y_T, Y_P,
#                                                     average="micro")
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#      .format(average_precision["micro"]))
#
############################
#
#plt.figure()
#plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
#         where='post')
#plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
#                 color='b')
#
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title(
#    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
#    .format(average_precision["micro"]))
#
#
######################################
#
#
#from itertools import cycle
## setup plot details
#colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#
#plt.figure(figsize=(7, 8))
#f_scores = np.linspace(0.2, 0.8, num=4)
#lines = []
#labels = []
#for f_score in f_scores:
#    x = np.linspace(0.01, 1)
#    y = f_score * x / (2 * x - f_score)
#    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
#
#lines.append(l)
#labels.append('iso-f1 curves')
#l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
#lines.append(l)
#labels.append('micro-average Precision-recall (area = {0:0.2f})'
#              ''.format(average_precision["micro"]))
#
#for i, color in zip(range(n_classes), colors):
#    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
#    lines.append(l)
#    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
#                  ''.format(i, average_precision[i]))
#
#fig = plt.gcf()
#fig.subplots_adjust(bottom=0.25)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Extension of Precision-Recall curve to multi-class')
#plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
#
#
#plt.show()
#
#
