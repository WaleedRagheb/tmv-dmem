# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:45:35 2018

@author: HP
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix



def computeP_R_F1(CM):
    TP = np.diag(CM)
    FP = np.sum(CM, axis=0) - TP
    FN = np.sum(CM, axis=1) - TP
    
    num_classes = 2
    TN = []
    for i in range(num_classes):
        temp = np.delete(CM, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))
    
    l = 401
#    for i in range(num_classes):
#        print(TP[i] + FP[i] + FN[i] + TN[i] == l)
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*((precision * recall)/(precision + recall))
    return precision, recall , F1



x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_test  = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[10], dtype=np.str)

random_state = np.random.RandomState(0)

classifier = svm.LinearSVC(class_weight={"POS":1.5})
classifier.fit(x_train, y_train)
y_score = classifier.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("Support Vector Machine : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_score = gnb.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("GaussianNB : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
P,R,F1 = computeP_R_F1(M)
print("Stochastic Gradient Descent : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")



from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("NearestCentroid : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("Decision Trees : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)

print("RandomForestClassifier : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")



model = SelectFromModel(clf, prefit=True)
X_new = model.transform(x_train)
print(X_new.shape)


selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(x_train, y_train)
idxs_selected = selector.get_support(indices=True)
print(idxs_selected)

x_train_new = x_train[:,0:3]
x_test_new = x_test[:,0:3]
clf.fit(x_train_new, y_train)
y_score = clf.predict(x_test_new)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)

print("RandomForestClassifier FS : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")



from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("ExtraTreesClassifier : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")



from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth=2, random_state=0)
clf.fit(x_train, y_train)
y_score = clf.predict(x_test)
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("GradientBoostingClassifier : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,100), random_state=1,activation='tanh')
#clf.n_outputs_ = 2 
clf.fit(x_train[:,0:8], y_train)
y_score = clf.predict(x_test[:,0:8])
M = confusion_matrix(y_test, y_score)
P,R,F1 = computeP_R_F1(M)
print("MLP : ")
print(M)
print("P = " + str(P) + "\nR = " + str(R) + "\nF1 = " + str(F1) + "\n-------------")


#average_precision = average_precision_score(y_test, y_score)

#print('Average precision-recall score: {0:0.2f}'.format(average_precision))