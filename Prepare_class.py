from fastai.text import *
from AWDLSTM.create_toks_2 import *
import html
from AWDEncClas.bwd_ids_transformer import create_bw_data_class



#CLAS_PATH = "data/nlp_clas/imdb/imdb_clas"
#LM_PATH = "data/nlp_clas/imdb"

#CLAS_PATH = "data/nlp_clas/yelp_bi/yelp_bi_clas"
#LM_PATH = "data/nlp_clas/yelp_bi/yelp_bi_lm"


#CLAS_PATH = "data/nlp_clas/yelp_full/yelp_full_clas"
#LM_PATH = "data/nlp_clas/yelp_full/yelp_full_lm"


#CLAS_PATH = "data/nlp_clas/eRisk_dep/eRisk_dep_clas"
#LM_PATH = "data/nlp_clas/eRisk_dep/eRisk_dep_lm"

#CLAS_PATH = "data/nlp_clas/amazon_full/amazon_full_clas"
#LM_PATH = "data/nlp_clas/amazon_full/amazon_full_lm"

CLAS_PATH = "data/nlp_clas/amazon_bi/amazon_bi_clas"
LM_PATH = "data/nlp_clas/amazon_bi/amazon_bi_lm"

chunksize = 24000

df_trn = pd.read_csv(CLAS_PATH+'/'+'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH+'/'+'test.csv', header=None, chunksize=chunksize)

tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)

np.save(CLAS_PATH+'/'+'tmp'+'/'+'tok_trn.npy', tok_trn)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tok_val.npy', tok_val)

np.save(CLAS_PATH+'/'+'tmp'+'/'+'lbl_trn.npy', trn_labels)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'lbl_val.npy', val_labels)

tok_trn = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tok_trn.npy')
tok_val = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tok_val.npy')
itos = pickle.load(open(LM_PATH+'/'+'tmp'+'/'+'itos.pkl','rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)

trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])
np.save(CLAS_PATH+'/'+'tmp'+'/'+'trn_ids.npy', trn_clas)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'val_ids.npy', val_clas)

create_bw_data_class(CLAS_PATH)