import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from fastai.text import *
from AWDEncClas.eval_clas import *
from AWDLSTM.create_toks_2 import *



PATH = Path('/home/waleed/data/eRisk/eRisk_Dep/eRisk_T1')
CLAS_PATH = 'data/nlp_clas/eRisk_dep/eRisk_dep_clas'
LM_PATH = "data/nlp_clas/eRisk_dep/eRisk_dep_lm"
#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
#LM_PATH.mkdir(exist_ok=True)
CLASSES = ['neg', 'pos']

def get_texts(path):
    texts,labels,subjID, chunckID = [],[], [], []
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            fileName = os.path.splitext(os.path.basename(fname))[0]
            sID,chNum = fileName.split("_")
            txt = fname.open('r', encoding='utf-8').read()
            #towri = txt.replace("~~", " ").replace("\"", "\"\"").replace("||", " ").replace("\n", " ").replace("\r","")
            texts.append(txt)
            labels.append(idx)
            subjID.append(sID)
            chunckID.append(chNum)
    return np.array(texts),np.array(labels), subjID, chunckID


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
    f = metrics[2]
    return f[len(f) - 1]





#trn_texts,trn_labels = get_texts(PATH/'train')
val_texts, val_labels, subjIDs, chunkIDs = get_texts(PATH/'test')

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
writeToCSV_oneFiled(df_val, CLAS_PATH+'/test_Real3.csv')


chunksize = 24000


df_val = pd.read_csv(CLAS_PATH+'/'+'test_Real3.csv', header=None, chunksize=24000)#,names=(range(227)))


tok_val, val_labels = get_all_eRisk(df_val, 1)


np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy', tok_val)


np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'lbl_val.npy', val_labels)


tok_val = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'tok_val.npy')
itos = pickle.load(open(LM_PATH+'/'+'tmp'+'/'+'itos.pkl','rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)


val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

np.save(CLAS_PATH+'/'+'tmp'+'/'+'tests'+'/'+'val_ids.npy', val_clas)

#prediction = eval_clas_eRisk(str(CLAS_PATH), 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4', attention=False)

prediction, samplIdx, dump = eval_clas_eRisk(str(CLAS_PATH), 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4_3', attention=False)





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

np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep4_3.npy', prediction)
np.save(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep4_3.npy', samplIdx)

prediction = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'prediction_eRisk_dep4_3.npy')
samplIdx = np.load(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'samplIdx_eRisk_dep4_3.npy')

subjIDs = [subjIDs[i] for i in samplIdx]
val_labels = [val_labels[i] for i in samplIdx]
chunkIDs = [chunkIDs[i] for i in samplIdx]


uniq_subID = list(set(subjIDs))
#results_pred = [[ None ] * 10] * len(uniq_subID)
results_pred = [[None]*10 for _ in range(len(uniq_subID))]
results_golden = [ None ] * len(uniq_subID)

#checkIntegrity(subjIDs,val_labels)

for sid,vLbls,chNum,pred in zip(subjIDs,val_labels,chunkIDs,prediction):
    idx = uniq_subID.index(sid)
    if results_golden[idx] != vLbls and  results_golden[idx] is not None:
        pass
    else:
        results_golden[idx] = vLbls
    results_pred[idx][int(chNum)-1] = pred

print("F1: " + str(f1_erisk(val_labels, prediction)))

All_df = pd.DataFrame({'Subject_ID':uniq_subID, 'results_pred':results_pred, 'results_golden':results_golden})
All_df.to_csv(str(CLAS_PATH)+'/'+'tmp/tests'+'/'+'Results4_3.csv', header=False, index=False)

