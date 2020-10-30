# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:34:14 2018

@author: HP
"""




import numpy as np
from sklearn.neural_network import MLPClassifier



x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])#,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/What_If/testingDataSet_ch10.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])#,1,2,3,4,5,6,7,8,9])


clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,100), random_state=1,activation='tanh')


clf_MLP.fit(x_train, y_train)

y_score = clf_MLP.predict(x_test)
y_proba = clf_MLP.predict_proba(x_test)

#towrite= np.empty((820, 2))
#towrite[:,0] = y_score
#towrite[:,1] = y_proba[:,1]

np.savetxt("./MLResults/What_If/testingDataSet_ch10_MLP.csv", y_proba[:,1].astype(str), delimiter=",", fmt='%10s',header="MLP")
