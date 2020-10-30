# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:56:39 2018

@author: HP
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier



###############################################
##############################################
def runClassifier(clf,x_train,y_train,x_test):
    
    
    x_train_sub = x_train
    x_test_sub = x_test
          
    clf.fit(x_train_sub, y_train)
#        y_score = clf.predict_proba(x_test_sub)
    y_score = clf.predict(x_test_sub)

#    return y_score[:,0]
    return y_score

    
##################################################
##################################################


x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2])#,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/RealTest/testingDataSet_ch3.csv", delimiter=",", usecols=[0,1,2])#,1,2,3,4,5,6,7,8,9])


clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 2), random_state=1)
#clf_SVM = svm.LinearSVC()
clf_GNB = GaussianNB()
clf_SGD = SGDClassifier(loss="modified_huber", penalty="l2")
#clf_NN = NearestCentroid()
clf_DT = tree.DecisionTreeClassifier()
clf_RF = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf_ETC = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth=2, random_state=0)
clf_Voting = VotingClassifier(estimators=[('mlp', clf_MLP), ('GNB', clf_GNB),('SGD', clf_SGD),  ('DT', clf_DT),('RF', clf_RF), ('ETC', clf_ETC), ('GBC', clf_GBC)], voting='soft')


allClassifiers = {"MLP": clf_MLP,
#                  "SVM": clf_SVM,
                  "GNB": clf_GNB,
                  "SGD": clf_SGD,
#                  "NN": clf_NN,
                  "DT": clf_DT,
                  "RF": clf_RF,
                  "ETC": clf_ETC,
                  "GBC": clf_GBC,
                  "VOTE": clf_Voting}


arrch1 = np.empty((820, 8), dtype=object)
m_ctr = 0

Modelidx = 0
for k,v in allClassifiers.items():
    # row is a list of lists
    print(k)
    arrch1[:,m_ctr] = runClassifier(v,x_train,y_train,x_test)
    m_ctr += 1


np.savetxt("./MLResults/RealTest/testingDataSet_ch3_ResCat.csv", arrch1.astype(str), delimiter=",", fmt='%10s',header="MLP,GNB,SGD,DT,RF,ETC,GBC,Vote,Golden_truth")