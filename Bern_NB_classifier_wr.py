import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from fastai.text import *
from AWDEncClas.eval_clas import *
from AWDLSTM.create_toks_2 import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma
import math


def f1_erisk(targs, preds):
    print(confusion_matrix(targs,preds))
    metrics = precision_recall_fscore_support(targs, preds, average=None)
    p = metrics[0]
    r = metrics[1]
    f = metrics[2]
    print("p: " + str(p[len(f) - 1]) + "\tR: " + str(r[len(f) - 1]) + "\tF1: " + str(f[len(f) - 1]))
    return f[len(f) - 1]


def run_model(data, nb_iterations=15):
    """

    :param data: a list of list, [nb_users, nb_writings]
    :param nb_iterations:
    :return:
    """
    uk = np.zeros((len(data),))

    for i in range(len(data)):
        uk[i] = np.random.randint(0, 2)
    pi = np.zeros((2, 2))  # row 0 is negative and row 1 is positive, column 0 is negative and 1 positive
    # rows are categories of users and columns of writings

    N = np.zeros((2,))
    Nz = np.zeros((2, 2))

    kappa = np.zeros((2, ))

    alpha = 1
    beta = 1

    lam_p = 15
    gamma_p = 1

    lam_n = 1
    gamma_n = 15

    for it in range(nb_iterations):
        print("*"*10+ '['+str(it)+']')

        N[:] = 0
        for k in range(uk.shape[0]):
            N[1] += uk[k]
            N[0] += 1-uk[k]

        Nz[:] = 0
        for k in range(uk.shape[0]):
            for i in range(len(data[k])):
                Nz[0, 0] += (1-data[k][i]) * (1 - uk[k])
                Nz[0, 1] += (data[k][i]) * (1 - uk[k])
                Nz[1, 0] += (1 - data[k][i]) * (uk[k])
                Nz[1, 1] += (data[k][i]) * (uk[k])

        pi[0, 0] = digamma(gamma_n + Nz[0, 0]) - digamma(gamma_n + Nz[0, 0] + lam_n + Nz[0, 1])
        pi[0, 1] = digamma(lam_n + Nz[0, 1]) - digamma(gamma_n + Nz[0, 0] + lam_n + Nz[0, 1])
        pi[1, 0] = digamma(gamma_p + Nz[1, 0]) - digamma(gamma_p + Nz[1, 0] + lam_p + Nz[1, 1])
        pi[1, 1] = digamma(lam_p + Nz[1, 1]) - digamma(gamma_p + Nz[1, 0] + lam_p + Nz[1, 1])


        kappa[0] = digamma(alpha + N[1]) - digamma(alpha + beta + N[1] + N[0])
        kappa[1] = digamma(beta + N[0]) - digamma(alpha + beta + N[1] + N[0])

        for k in range(uk.shape[0]):
            rho_0 = (beta - 1) * kappa[0]
            rho_1 = (alpha - 1) * kappa[1]
            for i in range(len(data[k])):
                rho_0 += data[k][i] * pi[0, 1] + (1 - data[k][i]) * pi[0, 0]
                rho_1 += data[k][i] * pi[1, 1] + (1 - data[k][i]) * pi[1, 0]

            if(math.isnan(np.exp(rho_1) / (np.exp(rho_1) + np.exp(rho_0)))):
                uk[k] = 0
            else:
                uk[k] = np.exp(rho_1) / (np.exp(rho_1) + np.exp(rho_0))
    return uk



#   print(run_model(data, nb_iterations=500))


if __name__ == '__main__':
    CLAS_PATH = 'data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class'
    df_val_ALL_OUT = pd.read_pickle(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'df_val_ALL_OUT.pkl')
    subj_list = set([it[0] for it in df_val_ALL_OUT.loc[:, ['subj_ID']].values.tolist()])

    data = []
    G_truth =[]
    for sId in subj_list:
        df_subj = df_val_ALL_OUT.loc[df_val_ALL_OUT['subj_ID'] == sId]
        wr_prob = [1 if(it[0]>=0.8) else 0 for it in df_subj.loc[:, ['pos_prob']].values.tolist()]
        data.append(wr_prob)
        G_truth.append([it[0] for it in df_subj.loc[:, ['labels']].values.tolist()][0])

    #data2 = [
    #  [0, 0, 1, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 1, 0, 0],
    #  [1, 0, 1, 1, 0, 1, 0, 1, 1],
    #  [1, 1, 1, 1, 0, 1],
    #  [1, 0, 1, 0, 1, 1],
    #  [1, 1, 1, 0, 1, 1],
    #  [1, 0, 1, 1, 1, 1],
    #  [1, 0, 1, 0, 1, 0, 1, 0, 1]
    # ]




    pred_inf = list(run_model(data, nb_iterations=20))

    th=0.6
    pred_inf_bi = [1 if it >=th else 0 for it in pred_inf]

    f1_erisk(G_truth, pred_inf_bi)












