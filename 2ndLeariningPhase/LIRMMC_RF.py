# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:18:54 2018

@author: HP
"""



import numpy as np
from sklearn.ensemble import RandomForestClassifier



x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])#,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/RealTest/testingDataSet_ch10.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])#,1,2,3,4,5,6,7,8,9])


clf_RF = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)


clf_RF.fit(x_train, y_train)

y_score = clf_RF.predict(x_test)
y_proba = clf_RF.predict_proba(x_test)

#towrite= np.empty((820, 2))
#towrite[:,0] = y_score
#towrite[:,1] = y_proba[:,1]

np.savetxt("./MLResults/RealTest/testingDataSet_ch10_RF.csv", y_proba[:,1].astype(str), delimiter=",", fmt='%10s',header="RF")