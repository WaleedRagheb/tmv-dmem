from fastai.text import *
import html

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

#PATH = Path('/home/waleed/data/imdb/imdb2/aclImdb/')
#CLAS_PATH = Path('data/nlp_clas/imdb/imdb_clas')
#CLAS_PATH.mkdir(exist_ok=True)
#
#LM_PATH = Path('data/nlp_clas/imdb/')
#LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos', 'unsup']



#PATH = Path('/home/waleed/data/eRisk/eRisk_Dep/eRisk_T1')
#CLAS_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_clas')
#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_dep/eRisk_dep_lm')
#LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']


#PATH = Path('/home/waleed/data/eRisk/eRisk_anox')
#CLAS_PATH = Path('data/nlp_clas/eRisk_anx/eRisk_anx_clas')
#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_anx/eRisk_anx_lm')
#LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']



#PATH = Path('/home/waleed/data/eRisk/eRisk_Dep_wr')
#CLAS_PATH = Path('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class')
#CLAS_PATH.mkdir(exist_ok=True)
#LM_PATH = Path('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm')
#LM_PATH.mkdir(exist_ok=True)
#CLASSES = ['neg', 'pos']

PATH = Path('/home/waleed/data/eRisk/eRisk_anx_wr')
CLAS_PATH = Path('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class')
CLAS_PATH.mkdir(exist_ok=True)
LM_PATH = Path('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_lm')
LM_PATH.mkdir(exist_ok=True)
CLASSES = ['neg', 'pos']


def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    #return np.array(texts),np.array(labels)
    return texts, labels


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


col_names = ['labels','text']
np.random.seed(42)

trn_texts,trn_labels = get_texts(PATH/'train')
trn_idx = np.random.permutation(len(trn_texts))
#rn_texts = trn_texts[trn_idx]
trn_texts = [trn_texts[i] for i in trn_idx]
#trn_labels = trn_labels[trn_idx]
trn_labels = [trn_labels[i] for i in trn_idx]
df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
writeToCSV_oneFiled(df_trn, CLAS_PATH/'train.csv')


val_texts,val_labels = get_texts(PATH/'test')
val_idx = np.random.permutation(len(val_texts))
#val_texts = val_texts[val_idx]
val_texts = [val_texts[i] for i in val_idx]
#val_labels = val_labels[val_idx]
val_labels = [val_labels[i] for i in val_idx]
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)
writeToCSV_oneFiled(df_val, CLAS_PATH/'test.csv')


#df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)
#df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)
#
#(CLAS_PATH/'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)
#
##trn_texts, val_texts = sklearn.model_selection.train_test_split(
##    np.concatenate([trn_texts,val_texts]), test_size=0.1)
#
#df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
#df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)
#
#df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
#df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)

