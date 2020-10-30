from fastai.text import *
from AWDLSTM.create_toks_2 import *
from AWDEncClas.bwd_ids_transformer import create_bw_data_class
import html



#CLAS_PATH = "data/nlp_clas/imdb/imdb_clas"
#LM_PATH = "data/nlp_clas/imdb"

#CLAS_PATH = "data/nlp_clas/yelp_bi/yelp_bi_clas"
#LM_PATH = "data/nlp_clas/yelp_bi/yelp_bi_lm"


#CLAS_PATH = "data/nlp_clas/yelp_full/yelp_full_clas"
#LM_PATH = "data/nlp_clas/yelp_full/yelp_full_lm"


#CLAS_PAT#H = "data/nlp_clas/eRisk_dep/eRisk_dep_clas"
#LM_PATH = "data/nlp_clas/eRisk_dep/eRisk_dep_lm"


CLAS_PATH = "data/nlp_clas/eRisk_anx/eRisk_anx_clas"
LM_PATH = "data/nlp_clas/eRisk_anx/eRisk_anx_lm"

#CLAS_PATH = "data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class"
#LM_PATH = "data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm"


#CLAS_PATH = "data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class"
#LM_PATH = "data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_lm"

chunksize = 24000

df_trn = pd.read_csv(CLAS_PATH+'/'+'train.csv', header=None, chunksize=chunksize)#,names=(range(233)))
df_val = pd.read_csv(CLAS_PATH+'/'+'test.csv', header=None, chunksize=chunksize)#,names=(range(227)))

tok_trn, trn_labels = get_all_eRisk(df_trn, 1)
tok_val, val_labels = get_all_eRisk(df_val, 1)

np.save(CLAS_PATH+'/'+'tmp'+'/'+'tok_trn.npy', tok_trn)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'tok_val.npy', tok_val)

np.save(CLAS_PATH+'/'+'tmp'+'/'+'lbl_trn.npy', trn_labels)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'lbl_val.npy', val_labels)

tok_trn = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tok_trn.npy')
tok_val = np.load(CLAS_PATH+'/'+'tmp'+'/'+'tok_val.npy')
itos = pickle.load(open(CLAS_PATH+'/'+'tmp'+'/'+'itos.pkl','rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
print(len(itos))

trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
val_clas = np.array([[stoi[o] for o in p] for p in tok_val])
np.save(CLAS_PATH+'/'+'tmp'+'/'+'trn_ids.npy', trn_clas)
np.save(CLAS_PATH+'/'+'tmp'+'/'+'val_ids.npy', val_clas)


create_bw_data_class(CLAS_PATH)