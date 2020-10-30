import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from fastai.text import *
from AWDEncClas.eval_clas import *
from AWDLSTM.create_toks_2 import *
import matplotlib
import matplotlib.pyplot as plt


#PATH = Path('/home/waleed/data/eRisk/eRisk_Dep_wr')
#CLAS_PATH = 'data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class'
#LM_PATH = "data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm"
##CLAS_PATH.mkdir(exist_ok=True)
##LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
##LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']


PATH = Path('/home/waleed/data/eRisk/eRisk_anx_wr')
CLAS_PATH = 'data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class'

#LM_PATH = "data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_lm"

#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
#LM_PATH.mkdir(exist_ok=True)
CLASSES = ['neg', 'pos']

lm_id='eRisk_anx_FT2'
clas_id='eRisk_anx_3_BEST'

#CLAS_PATH_mx = 'data/nlp_clas/eRisk_anx/eRisk_anx_clas'
LM_PATH = "data/nlp_clas/eRisk_anx/eRisk_anx_lm"

def get_texts(path):
    texts,labels,subjID, chunckID, wrID = [],[], [], [], []
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            fileName = os.path.splitext(os.path.basename(fname))[0]
            sID, chNum, wrNum = fileName.split("_")
            txt = fname.open('r', encoding='utf-8').read()
            #towri = txt.replace("~~", " ").replace("\"", "\"\"").replace("||", " ").replace("\n", " ").replace("\r","")
            texts.append(txt)
            labels.append(idx)
            subjID.append(sID)
            chunckID.append(chNum)
            wrID.append(wrNum)
    return texts,labels, subjID, chunckID, wrID


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




def checkIntegrity(subjIDs,val_labels):
    uniqueSubj = list(set(subjIDs))
    c = []
    for sub in uniq_subID:
        indices = [i for i, x in enumerate(subjIDs) if x == sub]
        labels = [int(val_labels[ind]) for ind in indices]
        uniqLab = set(list(labels))
        if len(uniqLab) > 1:
            c.append(sub)
    #print(c)
    return c



def f1_erisk(targs, preds):
    print(confusion_matrix(targs,preds))
    metrics = precision_recall_fscore_support(targs, preds, average=None)
    p = metrics[0]
    r = metrics[1]
    f = metrics[2]
    print("p: " + str(p[len(f) - 1]) + "\tR: " + str(r[len(f) - 1]) + "\tF1: " + str(f[len(f) - 1]))
    return f[len(f) - 1]


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






val_texts, val_labels, subjIDs, chunkIDs, wrIDs = get_texts(PATH/'test')


df_val_ALL = pd.DataFrame({'text':val_texts, 'labels':val_labels,'chunk_No': list(map(int, chunkIDs)),'wr_No': list(map(int, wrIDs)), 'subj_ID': subjIDs})
df_val_ALL = df_val_ALL.sort_values(by=['subj_ID', 'chunk_No', 'wr_No'], ascending=[True, True, True])
subjIDs_unq = set(subjIDs)

df_val_ALL_proc = df_val_ALL[0:0]
for sId in subjIDs_unq:
    print(sId)
    df_subj = df_val_ALL.loc[df_val_ALL['subj_ID'] == sId]
    df_subj_proc = ProcessAcc(df_subj,5)
    df_val_ALL_proc = df_val_ALL_proc.append(df_subj_proc)



val_texts = [it[0] for it in df_val_ALL_proc.loc[:, ['text']].values.tolist()]
val_labels = [it[0] for it in df_val_ALL_proc.loc[:, ['labels']].values.tolist()]
subjIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['subj_ID']].values.tolist()]
chunkIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['chunk_No']].values.tolist()]
wrIDs = [it[0] for it in df_val_ALL_proc.loc[:, ['wr_No']].values.tolist()]

np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_texts.npy', val_texts)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_labels.npy', val_labels)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'subjIDs.npy', subjIDs)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'chunkIDs.npy', chunkIDs)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'wrIDs.npy', wrIDs)

val_texts = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_texts.npy')
val_labels = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_labels.npy')
subjIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'subjIDs.npy')
chunkIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'chunkIDs.npy')
wrIDs = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'wrIDs.npy')

col_names = ['labels','text']


df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)
writeToCSV_oneFiled(df_val, CLAS_PATH+'/test_Real1_wr.csv')


chunksize = 24000


df_val1 = pd.read_csv(CLAS_PATH+'/'+'test_Real1_wr.csv', header=None, chunksize=chunksize, engine='python')#,names=(range(227)))


tok_val, val_labels = get_all_eRisk(df_val1, 1)


np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy', tok_val)


np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'lbl_val.npy', val_labels)


tok_val = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy')
itos = pickle.load(open(LM_PATH+'/'+'tmp'+'/'+'itos.pkl','rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)



val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_ids.npy', val_clas)

#prediction = eval_clas_eRisk(str(CLAS_PATH), 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4', attention=False)

prediction, samplIdx, pos_prob = eval_clas_eRisk(str(CLAS_PATH), 2, lm_id=lm_id, clas_id=clas_id, attention=False)



np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep_wr.npy', prediction)
np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep_wr.npy', samplIdx)
np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'pos_prob_eRisk_dep_wr.npy', pos_prob)

prediction = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep_wr.npy')
samplIdx = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep_wr.npy')
pos_prob = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'pos_prob_eRisk_dep_wr.npy')


subjIDs = [subjIDs[i] for i in samplIdx]
val_labels = [val_labels[i] for i in samplIdx]
chunkIDs = [chunkIDs[i] for i in samplIdx]
wrIDs = [wrIDs[i] for i in samplIdx]


uniq_subID = list(set(subjIDs))
#results_pred = [[ None ] * 10] * len(uniq_subID)
results_pred = [[0]*10 for _ in range(len(uniq_subID))]
results_pred_thr = [[0]*10 for _ in range(len(uniq_subID))]
totalWrs = [[0]*10 for _ in range(len(uniq_subID))]
results_golden = [ None ] * len(uniq_subID)

pos_prob_wr = [ [] for _ in range(len(uniq_subID)) ]

#checkIntegrity(subjIDs,val_labels)
th = 0.9



for sid,vLbls,chNum,wr, pred, pos_p in zip(subjIDs,val_labels,chunkIDs,wrIDs, prediction, pos_prob):
    idx = uniq_subID.index(sid)
    if results_golden[idx] != vLbls and  results_golden[idx] is not None:
        pass
    else:
        results_golden[idx] = vLbls
        pos_prob_wr[idx].append(pos_p)
    results_pred[idx][int(chNum)-1] = results_pred[idx][int(chNum)-1] + pred

    if(pos_p >= th):
        results_pred_thr[idx][int(chNum) - 1] = results_pred_thr[idx][int(chNum) - 1] + 1

    totalWrs[idx][int(chNum)-1] = max(totalWrs[idx][int(chNum)-1], int(wr))

#res = [[res/total for res,total in zip(results_pred[i], totalWrs[i])] for i in range(len(results_pred))]

print("F1: " + str(f1_erisk(val_labels, prediction)))

pos_p_avg = [sum(pb)/len(pb) for pb in pos_prob_wr]

#for thr in np.arange(0.0, 1.0, 0.025):
#    print(str(thr) + ":\t")
#    prd = [1 if(it >= thr) else 0 for it in pos_p_avg]
#    #print("F1: " + str(f1_erisk(prd, results_golden)))
#    f1_erisk(prd, results_golden)
#

pred_count = [sum(rb) for rb in results_pred_thr]
for thr in np.arange(0, 20, 1):
    print(str(thr) + ":\t")
    prd = [1 if(it >= thr) else 0 for it in pred_count]
    #print("F1: " + str(f1_erisk(prd, results_golden)))
    f1_erisk(prd, results_golden)

All_df = pd.DataFrame({'Subject_ID':uniq_subID, 'results_pred':results_pred, 'results_golden':results_golden})
All_df.to_csv(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'Results1_wr.csv', header=False, index=False)


prd_5 = [1 if(it >= 5) else 0 for it in pred_count]
df_val_ALL_OUT = pd.DataFrame({'text':val_texts, 'labels':val_labels,'chunk_No': list(map(int, chunkIDs)),'wr_No': list(map(int, wrIDs)), 'subj_ID': subjIDs, 'pos_prob': pos_prob})
df_val_ALL_OUT = df_val_ALL_OUT.sort_values(by=['subj_ID', 'chunk_No', 'wr_No'], ascending=[True, True, True])
df_val_ALL_OUT.to_pickle(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'df_val_ALL_OUT.pkl')
subjIDs_unq = set(subjIDs)
for i, sId in enumerate(subjIDs_unq):
    print(sId)
    df_subj = df_val_ALL_OUT.loc[df_val_ALL_OUT['subj_ID'] == sId]
    columnsData = [it[0] for it in df_subj.loc[:, ['pos_prob']].values.tolist()]
    lbl = int(df_subj.loc[:, ['labels']].values.tolist()[0][0])
    plt.plot(columnsData)
    plt.xlabel(sid + " ---- Pred_results: " + str(prd_5[i]) )
    plt.savefig('./Figs/eRisk_anx/' + str(lbl) + '/' + sId+ '.png')
    plt.close()



#res_df = pd.DataFrame({'Subject_ID':uniq_subID, 'results_pred':res, 'results_golden':results_golden})
#res_df.to_csv(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'Results1_wr_res.csv', header=False, index=False)

