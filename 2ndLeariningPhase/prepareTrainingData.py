# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:50:10 2018

@author: HP
"""

import sys
sys.path.insert(0, r'..\ChunkDealer')

import chunksProcessor

import sys
sys.path.insert(0, r'..\DeepModeling')


import testScoring
import gensim
import numpy as np
import os

##############################################################

def pos_neg_count(prob,prob_Thr_Diff):
    pos_cntr = 0
    neg_cntr = 0
    for index, row in prob.iterrows():
        pos_Prob = row[0]
        neg_Prob = row[1]
        diff = np.abs((pos_Prob-neg_Prob))
        if diff >= prob_Thr_Diff:
            if pos_Prob > neg_Prob:
                pos_cntr = pos_cntr + 1
            else:
                neg_cntr = neg_cntr + 1
    return pos_cntr, neg_cntr
            
        
#        print(row['c1'], row['c2'])
        

##############################################################

def giveDecision(prob, prob_Thr_Diff):
    cnt_p,cnt_n = pos_neg_count(prob, prob_Thr_Diff)
    sum_P_N = cnt_p + cnt_n
    if sum_P_N == 0:
        return 0, 0
    return cnt_p/sum_P_N, cnt_n/sum_P_N

##############################################################
prob_Thr_Diff = 0.0
for numberOfChunks in range(1,11):
    if numberOfChunks == 0:
        continue
    for model_ChunkN in range(10,11):
        if model_ChunkN == 0:
            continue
           
        Positive_chunkDir = r'C:\Users\HP\Downloads\clpsych16-data\2018\training\task1\eRisk@CLEF2018-task1-releasedtrainingdata\eRisk 2018 - training\2017 train\positive_examples_anonymous_chunks'
        Negative_chunkDir = r'C:\Users\HP\Downloads\clpsych16-data\2018\training\task1\eRisk@CLEF2018-task1-releasedtrainingdata\eRisk 2018 - training\2017 train\negative_examples_anonymous_chunks'
        model_Positive_path = r'../DeepModeling/Models_arch_6-plus/Positive/Chunk_' + str(model_ChunkN) + r'_300_c_40.word2vec'
        model_Negative_path = r'../DeepModeling/Models_arch_6-plus/Negative/Chunk_' + str(model_ChunkN) + r'_300_c_40.word2vec'
        
        sDic_pos = chunksProcessor.process_chunk_N_NotAcc(Positive_chunkDir,numberOfChunks)
        sDic_neg = chunksProcessor.process_chunk_N_NotAcc(Negative_chunkDir,numberOfChunks)
        modelPositive = gensim.models.Word2Vec.load(model_Positive_path)
        modelNegative = gensim.models.Word2Vec.load(model_Negative_path)
        
        if not os.path.exists(r'.\TempRes\Positive'):
            os.makedirs(r'.\TempRes\Positive')
            
        if not os.path.exists(r'.\TempRes\Negative'):
            os.makedirs(r'.\TempRes\Negative')
        
        fileName_pos = r'.\TempRes\Positive\Initial Results_' + str(numberOfChunks) + '_' + str(model_ChunkN) + '.txt'
        fileName_neg = r'.\TempRes\Negative\Initial Results_' + str(numberOfChunks) + '_' + str(model_ChunkN) + '.txt'
                    
        
        for k, v_txt in sDic_pos.items():
            sents = testScoring.txtToSen(v_txt)
            sentsList = list(sents)
            if len(sentsList) < 1:
                with open(fileName_pos, "a") as f:
                    f.write(k + "\t None\n")
                continue
            prob = testScoring.docprob(sentsList,[modelPositive,modelNegative])
        #    d_subj should be 0,1,2
            cnt_p, cnt_n = giveDecision(prob, prob_Thr_Diff)
            
            with open(fileName_pos, "a") as f:
                f.write(k + "\t " + str(cnt_p) + "\t " + str(cnt_n) + "\n")
                
        ###################################################################
        
        
        for k, v_txt in sDic_neg.items():
            sents = testScoring.txtToSen(v_txt)
            sentsList = list(sents)
            if len(sentsList) < 1:
                with open(fileName_neg, "a") as f:
                    f.write(k + "\t None\n")
                continue
            prob = testScoring.docprob(sentsList,[modelPositive,modelNegative])
        #    d_subj should be 0,1,2
            cnt_p, cnt_n = giveDecision(prob, prob_Thr_Diff)
            
            with open(fileName_neg, "a") as f:
                f.write(k + "\t " + str(cnt_p) + "\t " + str(cnt_n) + "\n")

#######################################################################
                

from os import listdir
from os.path import isfile, join
import numpy as np
import re
import matplotlib.pyplot as plt


res_matrix_pos = np.zeros((10,10,83,2),float)
subj_ID_array = []
resultsPath =r'.\TempRes\Positive'
onlyfiles = [f for f in listdir(resultsPath) if isfile(join(resultsPath, f))]


for res_f_name in onlyfiles:
    print(res_f_name)
    m = re.search('_(.+?)\.',res_f_name)
    chunk_idx, model_idx = m.group(1).split('_')
    
    subj_ID_array = np.loadtxt(resultsPath + '\\' + res_f_name, delimiter='\t', usecols=(0), unpack=True,dtype=np.str)
    
    
    try:
        Pos_Neg_mtrx = np.loadtxt(resultsPath + '\\' + res_f_name, delimiter='\t', usecols=(1, 2), unpack=True).transpose()
    except IndexError:
        Pos_Neg_mtrx = np.zeros((83,2),float)
        with open(resultsPath + '\\' + res_f_name) as fi:
            l_ctr = 0
            for line in fi:
                fields = line.split('\t')
                if len(fields) > 2:
                    Pos_Neg_mtrx[l_ctr][0] = fields[1]
                    Pos_Neg_mtrx[l_ctr][1] = fields[2]
                l_ctr += 1
        
    res_matrix_pos[int(chunk_idx)-1,int(model_idx)-1,:,:] = Pos_Neg_mtrx
   
############################################################################

res_matrix_neg = np.zeros((10,10,403,2),float)
subj_ID_array = []
resultsPath =r'.\TempRes\Negative'
onlyfiles = [f for f in listdir(resultsPath) if isfile(join(resultsPath, f))]


for res_f_name in onlyfiles:
    print(res_f_name)
    m = re.search('_(.+?)\.',res_f_name)
    chunk_idx, model_idx = m.group(1).split('_')
    
    subj_ID_array = np.loadtxt(resultsPath + '\\' + res_f_name, delimiter='\t', usecols=(0), unpack=True,dtype=np.str)
    
    
    try:
        Pos_Neg_mtrx = np.loadtxt(resultsPath + '\\' + res_f_name, delimiter='\t', usecols=(1, 2), unpack=True).transpose()
    except IndexError:
        Pos_Neg_mtrx = np.zeros((403,2),float)
        with open(resultsPath + '\\' + res_f_name) as fi:
            l_ctr = 0
            for line in fi:
                fields = line.split('\t')
                if len(fields) > 2:
                    Pos_Neg_mtrx[l_ctr][0] = fields[1]
                    Pos_Neg_mtrx[l_ctr][1] = fields[2]
                l_ctr += 1
        
    res_matrix_neg[int(chunk_idx)-1,int(model_idx)-1,:,:] = Pos_Neg_mtrx
   
with open(r'.\TempRes\TrainingData.csv', 'a') as trainFile:
    for itr in range(402):
        lstToWrite = res_matrix_neg[:,9,itr,0]
        for ii in range(len(lstToWrite)):
            trainFile.write(str(lstToWrite[ii]) + ",")
        trainFile.write("NEG\n")
    for itr in range(82):
        lstToWrite = res_matrix_pos[:,9,itr,0]
        for ii in range(len(lstToWrite)):
            trainFile.write(str(lstToWrite[ii]) + ",")
        trainFile.write("POS\n")
    trainFile.close()
    
