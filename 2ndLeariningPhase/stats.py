# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:01:38 2018

@author: HP
"""




import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x_train = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])#,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[10], dtype=np.str)

X_embedded = TSNE(n_components=2).fit_transform(x_train[:,0:10])

print(X_embedded.shape)

y_train_bin  = np.where(y_train=="POS", 1, 0)
plt.figure(figsize=(10, 5))

POS_Idx = np.where(y_train=="POS")
NEG_Idx = np.where(y_train=="NEG")

plt.scatter(np.array(X_embedded[:,0])[POS_Idx], np.array(X_embedded[:,1])[POS_Idx], marker = 'o' , label="Positive")
plt.scatter(np.array(X_embedded[:,0])[NEG_Idx], np.array(X_embedded[:,1])[NEG_Idx], marker = '^' , label="Negative")


plt.legend(numpoints=1)

plt.savefig('tsne_train.eps', format='eps', dpi=1000)

for i in range(len(y_train))

