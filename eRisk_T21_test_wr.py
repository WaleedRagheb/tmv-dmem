import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from fastai.text import *
from AWDEncClas.eval_clas import *
from AWDLSTM.create_toks_2 import *



#PATH = Path('/home/waleed/data/eRisk/eRisk_Dep_wr')
#CLAS_PATH = 'data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class'
#LM_PATH = "data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm"
##CLAS_PATH.mkdir(exist_ok=True)
##LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
##LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']


PATH = Path('/home/waleed/data/eRisk/eRisk_anx_wr')
CLAS_PATH = 'data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class'
LM_PATH = "data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_lm"
#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
#LM_PATH.mkdir(exist_ok=True)
CLASSES = ['neg', 'pos']
lm_id='eRisk_anx_wr_FT'
clas_id='eRisk_anx_wr_FT'

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





#trn_texts,trn_labels = get_texts(PATH/'train')
val_texts, val_labels, subjIDs, chunkIDs, wrIDs = get_texts(PATH/'test')

col_names = ['labels','text']
#np.random.seed(42)
##trn_idx = np.random.permutation(len(trn_texts))
#val_idx = np.random.permutation(len(val_texts))
##trn_texts = trn_texts[trn_idx]
#val_texts = val_texts[val_idx]
##trn_labels = trn_labels[trn_idx]
#val_labels = val_labels[val_idx]

#df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
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

prediction, samplIdx, pos_prob = eval_clas_eRisk(str(CLAS_PATH), 3, lm_id=lm_id, clas_id=clas_id, attention=False)





#tok_val, val_labels = get_all_eRisk(df_val, 1)
##tok_val = np.array(tok_val)
##val_labels = np.array(val_labels)
#
#
#np.save(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy', tok_val)
#np.save(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'lbl_val.npy', val_labels)
#
#
#tok_val = np.load(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy')
#
#itos = pickle.load(open(str(CLAS_PATH) + '/' + 'tmp' + '/' + 'itos.pkl', 'rb'))
#stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
#print(len(itos))
#val_clas_tok = np.array([[stoi[o] for o in p] for p in tok_val])
#np.save(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'val_ids.npy', val_clas_tok)
#
#val_clas_tok = np.load(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'val_ids.npy')
#val_labels = np.load(str(CLAS_PATH)+'/'+'tmp'+'/'+'tests'+'/'+'lbl_val.npy').flatten()
#
#prediction = eval_clas_eRisk(str(CLAS_PATH), 1, lm_id='eRisk_dep3_FT', clas_id='eRisk_dep3_14', attention=False, validIDs=val_clas_tok, validLbls=val_labels)
#

np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep_wr.npy', prediction)
np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep_wr.npy', samplIdx)

prediction = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep_wr.npy')
samplIdx = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep_wr.npy')

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
th = 0.7

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
for thr in np.arange(0, 100, 5):
    print(str(thr) + ":\t")
    prd = [1 if(it >= thr) else 0 for it in pred_count]
    #print("F1: " + str(f1_erisk(prd, results_golden)))
    f1_erisk(prd, results_golden)

All_df = pd.DataFrame({'Subject_ID':uniq_subID, 'results_pred':results_pred, 'results_golden':results_golden})
All_df.to_csv(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'Results1_wr.csv', header=False, index=False)

#res_df = pd.DataFrame({'Subject_ID':uniq_subID, 'results_pred':res, 'results_golden':results_golden})
#res_df.to_csv(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'Results1_wr_res.csv', header=False, index=False)

