import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import pandas as pd
from AWDLSTM.create_toks_2 import *
from AWDEncClas.eval_clas import *
from AWDEncClas.bwd_ids_transformer import create_bw_data_test





#classPath = "data/nlp_clas/imdb/imdb_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'imdb_FT_ATT_ML'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'imdb_FT'
#att=True

#classPath = "data/nlp_clas/imdb/imdb_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'imdb_FT_ATT_4rmScr'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'imdb_FT'
#att=True


#classPath = "data/nlp_clas/yelp_bi/yelp_bi_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'yelp_FT_ATT2_4rmScr_BEST'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'yelp_FT'
#att=True

#classPath = "data/nlp_clas/yelp_full/yelp_full_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'yelp_FT_ATT_14_4rmScr'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'yelpFull_FT'
#att=True

#classPath = "data/nlp_clas/amazon_full/amazon_full_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'amazonFull_FT_2'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'amazon_full_FT'
#att=False


#classPath = "data/nlp_clas/amazon_bi/amazon_bi_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'amazonBi_FT_ATT_BEST'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'amazon_bi_FT'
#att=True


#classPath = "data/nlp_clas/amazon_bi/amazon_bi_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'amazonBi_FT_ATT_BEST'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'amazon_bi_FT'
#att=True


#classPath = "data/nlp_clas/amazon_bi/amazon_bi_clas"
##LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
#classModelName = 'amazonBi_FT_ATT_4rmScr'
##classModelName = 'ph1Class_ws_n_BEST_1'
#LMModelName = 'amazon_bi_FT'
#att=True



classPath = "data/nlp_clas/amazon_full/amazon_full_clas"
#LMPath = "data/nlp_clas/train_ALL/train_ALL_lm"
classModelName = 'amazonFull_FT_ATT_4rmScr'
#classModelName = 'ph1Class_ws_n_BEST_1'
LMModelName = 'amazon_full_FT'
att=True

def generateTestOut(classPath, classModelName, LMModelName):








    prediction_fw, samplIdx_fw, val_lbls = eval_clas_ens(classPath, 2, lm_id=LMModelName, clas_id=classModelName, attention=att, bs=64, backwards=False, scores=True)
    prediction_bw, samplIdx_bw, val_lbls = eval_clas_ens(classPath, 2, lm_id=LMModelName, clas_id=classModelName,
                                                 attention=att, bs=64, backwards=True, scores=True)

    prediction_all =  [np.argmax(((softmax(p_fw))+(softmax(p_bw)))/2, axis=0) for p_fw, p_bw in zip(prediction_fw,prediction_bw)]

    acc = (val_lbls == prediction_all).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls, prediction_all))







generateTestOut(classPath, classModelName, LMModelName)
