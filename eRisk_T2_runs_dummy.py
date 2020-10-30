import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from fastai.text import *
from AWDEncClas.eval_clas import *
from AWDLSTM.create_toks_2 import *
import matplotlib
import matplotlib.pyplot as plt
from AWDEncClas.bwd_ids_transformer import create_bw_data_test
import gc


#PATH = Path('/home/waleed/data/eRisk/eRisk_Dep_wr')
#CLAS_PATH = 'data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class'
#LM_PATH = "data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm"
##CLAS_PATH.mkdir(exist_ok=True)
##LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
##LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']


PATH = Path('/home/waleed/data/eRisk/eRisk_anx_wr')
CLAS_PATH = 'data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class'


CLASSES = ['neg', 'pos']

lm_id='eRisk_anx_FT2'
clas_id='eRisk_anx_3_BEST'

LM_PATH = "data/nlp_clas/eRisk_anx/eRisk_anx_lm"

def get_texts_final(path):
    texts,subjID, chunckID, wrID = [],[], [], []

    for fname in (path).glob('*.*'):
        fileName = os.path.splitext(os.path.basename(fname))[0]
        sID, chNum, wrNum = fileName.split("_")
        txt = fname.open('r', encoding='utf-8').read()
        #towri = txt.replace("~~", " ").replace("\"", "\"\"").replace("||", " ").replace("\n", " ").replace("\r","")
        texts.append(txt)
        subjID.append(sID)
        chunckID.append(chNum)
        wrID.append(wrNum)
    return texts, subjID, chunckID, wrID


def writeToCSV(df_a, tarFile):
    with open(tarFile, 'w', encoding='utf-8') as trainF:
        for index, row in df_a.iterrows():
            if row['labels'] not in range(0, len(CLASSES)):
                print("Error.....!")
            trainF.write(str(row['labels']) + ",")
            text = str(row['text'])
            writings = text.split("~~")
            for wri in writings:
                towri = wri.replace("\"","\"\"").replace("||"," ").replace("\n", " ").replace("\r", "")
                trainF.write("\"" + towri + "\",")
            trainF.write("\n")
        trainF.close()


def writeToCSV_oneFiled(df_a, tarFile):
    with open(tarFile, 'w', encoding='utf-8') as trainF:
        for index, row in df_a.iterrows():
            if row['labels'] not in range(0, len(CLASSES)):
                print("Error.....!")
            trainF.write(str(row['labels']) + ",")
            text = str(row['text'])
            towri = text.replace("~~"," ").replace("\"", "\"\"").replace("||", " ").replace("\n", " ").replace("\r", "")
            trainF.write("\"" + towri + "\"\n")
        trainF.close()





#def f1_erisk(targs, preds):
#    print(confusion_matrix(targs,preds))
#    metrics = precision_recall_fscore_support(targs, preds, average=None)
#    p = metrics[0]
#    r = metrics[1]
#    f = metrics[2]
#    print("p: " + str(p[len(f) - 1]) + "\tR: " + str(r[len(f) - 1]) + "\tF1: " + str(f[len(f) - 1]))
#    return f[len(f) - 1]
#

def ProcessAcc(df_subj, window):
    df_subj = df_subj.reset_index(drop=True)
    df_subj_proc = df_subj[0:0]
    for index, row in df_subj.iterrows():
        startIdx = max(index - window, 0)
        columnsData = df_subj.loc[startIdx:index, ['text']].values.tolist()
        #print(columnsData)
        df_subj_proc.at[index,:] = df_subj.loc[index,:]
        df_subj_proc.at[index,'text'] = " ".join([it[0] for it in columnsData])

    return df_subj_proc



def ProcessAcc_opt(df_subj, window):
    df_subj = df_subj.reset_index(drop=True)
    df_subj_proc = df_subj[0:0]
    for index in df_subj.index:
        startIdx = max(index - window, 0)

        df_subj_proc.set_value(index,'text', " ".join([str(df_subj.get_value(i,'text')) for i in range (startIdx,index)]))
        df_subj_proc.set_value(index, 'chunk_No', df_subj.get_value(index,'chunk_No'))
        df_subj_proc.set_value(index, 'wr_No', df_subj.get_value(index, 'wr_No'))
        df_subj_proc.set_value(index, 'subj_ID', df_subj.get_value(index, 'subj_ID'))


    return df_subj_proc

def writeRunFile(runIdx,outDir, uniq_subID, prediction,scores):
    file = outDir + '/' + str(runIdx) + '.txt'
    with open(file, 'w') as filetowrite:
        for subj, p, s in zip(uniq_subID,prediction,scores):
            filetowrite.write(subj + '\t' + str(p) + '\t' + str(s) + '\n')
        filetowrite.close()


def logRun(runIdx,DecisionDir, uniq_subID, prediction, scores, maxRound):
    firstRound = 1 # we can change it to [zero] if we want to start from the initial sequence

    predFile = DecisionDir + '/predictionsHis_' + str(runIdx) + '.csv'
    scorFile = DecisionDir + '/scoresHis_' + str(runIdx) + '.csv'

    if maxRound == firstRound:
        df_p_toWrite = pd.DataFrame({'subjID': uniq_subID, 'pred': prediction})
        df_s_toWrite = pd.DataFrame({'subjID': uniq_subID, 'sc': scores})

        df_p_toWrite.to_csv(predFile, header=False, index=False)
        df_s_toWrite.to_csv(scorFile, header=False, index=False)
    else:

        df_p = pd.read_csv(predFile, header=None)
        df_s = pd.read_csv(scorFile, header=None)

        if (len(df_p.columns) - 1) == (maxRound-firstRound):
            newP = []
            for p_row in df_p[0]:
                newP.append(prediction[uniq_subID.index(p_row)])
            df_p[len(df_p.columns)] = newP

            newS = []
            for s_row in df_s[0]:
                newS.append(scores[uniq_subID.index(s_row)])
            df_s[len(df_s.columns)] = newS

            df_p.to_csv(predFile, header=False, index=False)
            df_s.to_csv(scorFile, header=False, index=False)




def createDecisionFiles(uniq_subID, pred_count, pos_prob_wr, DecisionDir, maxRound):
    thrs = [5, 6, 8, 10, 12]
    for runIdx , th in enumerate(thrs):
        prediction = [1 if (it >= th) else 0 for it in pred_count]
        scores = []
        for subjIdx, pos in enumerate(pos_prob_wr):
            p = [it for it in pos if it > 0.5]
            n = [1-it for it in pos if it <= 0.5]
            pred = prediction[subjIdx]
            if pred == 1:
                scores.append((sum(p) / (len(p)+1)) * 10)
            else:
                scores.append((sum(n) / (len(n)+1)) * 10)

        writeRunFile(runIdx+1, DecisionDir+'/last', uniq_subID, prediction, scores)
        logRun(runIdx+1, DecisionDir, uniq_subID, prediction, scores, maxRound)








def eRisk_dummy_runs(writingDir, DecisionDir):
    PATH = Path(writingDir)
    CLAS_PATH = 'data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class'

    CLASSES = ['neg', 'pos']

    lm_id = 'eRisk_anx_FT2'
    clas_id = 'eRisk_anx_3_BEST'

    LM_PATH = "data/nlp_clas/eRisk_anx/eRisk_anx_lm"

    val_texts, subjIDs, chunkIDs, wrIDs = get_texts_final(PATH)


    df_val_ALL = pd.DataFrame({'text':val_texts,'chunk_No': list(map(int, chunkIDs)),'wr_No': list(map(int, wrIDs)), 'subj_ID': subjIDs})
    df_val_ALL = df_val_ALL.sort_values(by=['subj_ID', 'chunk_No', 'wr_No'], ascending=[True, True, True])
    subjIDs_unq = set(subjIDs)

    df_val_ALL_proc = df_val_ALL[0:0]
    for sId in subjIDs_unq:
        #print(sId)
        df_subj = df_val_ALL.loc[df_val_ALL['subj_ID'] == sId]
        df_subj_proc = ProcessAcc_opt(df_subj,2)
        df_val_ALL_proc = df_val_ALL_proc.append(df_subj_proc)

    print('Finish [ProcessAcc].....' + str(max(map(int, wrIDs))))

    val_texts = [it[0] for it in df_val_ALL_proc.loc[:, ['text']].values.tolist()]
    #val_labels = [it[0] for it in df_val_ALL_proc.loc[:, ['labels']].values.tolist()]
    subjIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['subj_ID']].values.tolist()]
    chunkIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['chunk_No']].values.tolist()]
    wrIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['wr_No']].values.tolist()]

    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'val_texts.npy', val_texts)
    ##np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'val_labels.npy', val_labels)
    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'subjIDs.npy', subjIDs)
    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'chunkIDs.npy', chunkIDs)
    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'wrIDs.npy', wrIDs)
#
    #val_texts = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'val_texts.npy')
    ##val_labels = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'val_labels.npy')
    #subjIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'subjIDs.npy')
    #chunkIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'chunkIDs.npy')
    #wrIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'wrIDs.npy')

    col_names = ['labels','text']


    df_val = pd.DataFrame({0:np.zeros(len(val_texts)), 1:val_texts}) #columns=col_names)
    #writeToCSV_oneFiled(df_val, CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/test_Real1_wr.csv')


    #chunksize = 24000


    #df_val1 = pd.read_csv(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'test_Real1_wr.csv', header=None, chunksize=chunksize, engine='python')#,names=(range(227)))


    #tok_val, val_labels = get_all_eRisk(df_val1, 1)

    #tok_val, val_labels = get_all_eRisk_2(df_val, 1)

    tok_val, val_labels = get_texts_eRisk(df_val, 1)


    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'tok_val.npy', tok_val)


    #np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'lbl_val.npy', val_labels)


    #tok_val = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'tok_val.npy')
    itos = pickle.load(open(LM_PATH+'/'+'tmp'+'/'+'itos.pkl','rb'))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    len(itos)



    val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/'+'val_ids.npy', val_clas)

    #prediction = eval_clas_eRisk(str(CLAS_PATH), 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4', attention=False)

    create_bw_data_test(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'dummy'+'/', LM_PATH+'/'+'tmp'+'/'+'itos.pkl')

    prediction_fw, samplIdx_fw, pos_prob_fw = eval_clas_eRisk_final(str(CLAS_PATH), 2, lm_id=lm_id, clas_id=clas_id, attention=False)
    prediction_bw, samplIdx_bw, pos_prob_bw = eval_clas_eRisk_final(str(CLAS_PATH), 2, lm_id=lm_id, clas_id=clas_id, attention=False, backwards=True)

    prediction = [1 if (it_fw+it_bw)/2 > 0.5 else 0 for it_fw, it_bw in zip(prediction_fw, prediction_bw)]
    samplIdx = samplIdx_fw
    pos_prob = [(it_fw+it_bw)/2 for it_fw, it_bw in zip(pos_prob_fw, pos_prob_bw)]


    #np.save(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'prediction_eRisk_dep_wr_ens.npy', prediction)
    #np.save(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'samplIdx_eRisk_dep_wr_ens.npy', samplIdx)
    #np.save(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'pos_prob_eRisk_dep_wr_ens.npy', pos_prob)
#
    #prediction = np.load(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'prediction_eRisk_dep_wr_ens.npy')
    #samplIdx = np.load(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'samplIdx_eRisk_dep_wr_ens.npy')
    #pos_prob = np.load(str(CLAS_PATH)+'/'+'tmp/tests/dummy'+'/'+'pos_prob_eRisk_dep_wr_ens.npy')


    subjIDs = [subjIDs[i] for i in samplIdx]
    #val_labels = [val_labels[i] for i in samplIdx]
    chunkIDs = [chunkIDs[i] for i in samplIdx]
    wrIDs = [wrIDs[i] for i in samplIdx]


    uniq_subID = list(set(subjIDs))
    #results_pred = [[ None ] * 10] * len(uniq_subID)
    results_pred = [[0]*10 for _ in range(len(uniq_subID))]
    results_pred_thr = [[0]*10 for _ in range(len(uniq_subID))]
    totalWrs = [[0]*10 for _ in range(len(uniq_subID))]
    #results_golden = [ None ] * len(uniq_subID)

    pos_prob_wr = [ [] for _ in range(len(uniq_subID)) ]

    #checkIntegrity(subjIDs,val_labels)
    th = 0.9



    for sid,chNum,wr, pred, pos_p in zip(subjIDs,chunkIDs,wrIDs, prediction, pos_prob):
        idx = uniq_subID.index(sid)
        pos_prob_wr[idx].append(pos_p)
        results_pred[idx][int(chNum)-1] = results_pred[idx][int(chNum)-1] + pred

        if(pos_p >= th):
            results_pred_thr[idx][int(chNum) - 1] = results_pred_thr[idx][int(chNum) - 1] + 1

        totalWrs[idx][int(chNum)-1] = max(totalWrs[idx][int(chNum)-1], int(wr))

    #res = [[res/total for res,total in zip(results_pred[i], totalWrs[i])] for i in range(len(results_pred))]

    #print("F1: " + str(f1_erisk(val_labels, prediction)))

    pos_p_avg = [sum(pb)/len(pb) for pb in pos_prob_wr]

    #for thr in np.arange(0.0, 1.0, 0.025):
    #    print(str(thr) + ":\t")
    #    prd = [1 if(it >= thr) else 0 for it in pos_p_avg]
    #    #print("F1: " + str(f1_erisk(prd, results_golden)))
    #    f1_erisk(prd, results_golden)
    #

    pred_count = [sum(rb) for rb in results_pred_thr]

    createDecisionFiles(uniq_subID, pred_count, pos_prob_wr, DecisionDir, max(map(int, wrIDs)))
    gc.collect()


   # for thr in np.arange(0, 20, 1):
        #print(str(thr) + ":\t")
        #prd = [1 if(it >= thr) else 0 for it in pred_count]
        #print("F1: " + str(f1_erisk(prd, results_golden)))
        #f1_erisk(prd, results_golden)




if __name__ == "__main__":
    eRisk_dummy_runs('Java/Data/dummy/writings', 'Java/Data/dummy/Decisions')