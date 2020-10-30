# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:53:09 2018

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
    
#    l = 401
#    for i in range(num_classes):
#        print(TP[i] + FP[i] + FN[i] + TN[i] == l)
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*((precision * recall)/(precision + recall))
    return precision, recall , F1


###############################################
##############################################
def allRunsForClassifier(clf,x_train,y_train,x_test,y_test):
    ret_lst = []
    for ch in range(10):
        
        x_train_sub = x_train[:,0:ch+1]
        x_test_sub = x_test[:,0:ch+1]
              
        clf.fit(x_train_sub, y_train)
        y_score = clf.predict(x_test_sub)
        M = confusion_matrix(y_test, y_score)
        P,R,F1 = computeP_R_F1(M)
        sublist = [P[1], R[1], F1[1], y_score]
        ret_lst.append(sublist)
    
    return ret_lst

    
##################################################
##################################################
x_train = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_train  = np.genfromtxt(r"./MLResults/Train_Test/TrainingData.csv", delimiter=",", usecols=[10], dtype=np.str)

x_test = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[0,1,2,3,4,5,6,7,8,9])
y_test  = np.genfromtxt(r"./MLResults/Train_Test/TestingData.csv", delimiter=",", usecols=[10], dtype=np.str)



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


allClassifierResults = {"MLP": [],
#                  "SVM": [],
                  "GNB": [],
                  "SGD": [],
#                  "NN": [],
                  "DT": [],
                  "RF": [],
                  "ETC": [],
                  "GBC": [],}

arrch1 = np.empty((401, 8), dtype=object)
arrch2 = np.empty((401, 8), dtype=object)
arrch3 = np.empty((401, 8), dtype=object)
arrch4 = np.empty((401, 8), dtype=object)
arrch5 = np.empty((401, 8), dtype=object)
arrch6 = np.empty((401, 8), dtype=object)
arrch7 = np.empty((401, 8), dtype=object)
arrch8 = np.empty((401, 8), dtype=object)
arrch9 = np.empty((401, 8), dtype=object)
arrch10 = np.empty((401, 8), dtype=object)
allChunkResults = {"chunk-1": arrch1,
                   "chunk-2": arrch2,
                   "chunk-3": arrch3,
                   "chunk-4": arrch4,
                   "chunk-5": arrch5,
                   "chunk-6": arrch6,
                   "chunk-7": arrch7,
                   "chunk-8": arrch8,
                   "chunk-9": arrch9,
                   "chunk-10": arrch10,}

Modelidx = 0
for k,v in allClassifiers.items():
    # row is a list of lists
    print(k)
    row = allRunsForClassifier(v,x_train,y_train,x_test,y_test)
    allClassifierResults[k] = row
    
    for chIdx in range(10):
        print(chIdx)
        chRes = row[chIdx][3]
        allChunkResults["chunk-"+str(chIdx+1)][:,Modelidx] = chRes
    Modelidx += 1
        
        
        

chnkCtr = 0
for k,v in allChunkResults.items():
    allClassResForChunk = v
    withActual = np.empty((401, 9), dtype=object)
    withActual[:,0:8] = allClassResForChunk
    withActual[:,8] = y_test
    np.savetxt("./ML_models_over_chunks/Chunk-" + str(chnkCtr+1) + ".csv", withActual.astype(str), delimiter=",", fmt='%10s',header="MLP,GNB,SGD,DT,RF,ETC,GBC,Vote,Golden_truth")
    chnkCtr += 1
    
#print(allClassifierResults)


# predict class probabilities for all classifiers
probas = [c.fit(x_train, y_train).predict_proba(x_train) for c in (clf_MLP, clf_GNB, clf_SGD, clf_DT, clf_RF,clf_ETC, clf_GBC,clf_Voting )]

# get class probabilities for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]


# plotting
import matplotlib.pyplot as plt

#predict class probabilities for all classifiers


N = 8  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')

# bars for VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')

# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
plt.show()