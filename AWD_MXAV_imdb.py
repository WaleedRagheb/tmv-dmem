import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from AWDLSTM.create_toks_2 import *
from AWDLSTM.tok2id import *
from AWDEncClas.finetune_lm import*
from AWDEncClas.bwd_ids_transformer import create_bw_data

#create_toks("imdb","imdb_apr_cls", chunksize=24000, n_lbls=1)

#create_toks("yelp_bi","yelp_bi_apr_cls", chunksize=24000, n_lbls=1)

#create_toks("yelp_full","yelp_full_apr_cls", chunksize=24000, n_lbls=1)

#create_toks_eRisk("eRisk_dep", "eRisk_dep_apr_cls", chunksize=24000, n_lbls=1)

#create_toks("amazon_bi","amazon_bi_apr_cls", chunksize=24000, n_lbls=1)

#create_toks("amazon_full","amazon_bi_apr_cls", chunksize=24000, n_lbls=1)

#create_toks_eRisk("eRisk_anx", "eRisk_anx_apr_cls", chunksize=24000, n_lbls=1)

#create_toks_eRisk("eRisk_dep_wr", "eRisk_dep_wr_apr_cls", chunksize=24000, n_lbls=1)

#create_toks_eRisk("eRisk_anx_wr", "eRisk_anx_wr_apr_cls", chunksize=24000, n_lbls=1)

#tok2id("imdb", min_freq=2)

#tok2id("yelp_bi", min_freq=2)

#tok2id("yelp_full", min_freq=2)

#tok2id("eRisk_dep", min_freq=2)

#tok2id("amazon_bi", min_freq=2)

#tok2id("amazon_full", min_freq=2)

#tok2id("eRisk_anx", min_freq=2)

#tok2id("eRisk_dep_wr", min_freq=2)

#tok2id("eRisk_anx_wr", min_freq=2)

#create_bw_data('imdb')

#create_bw_data('yelp_bi')
#create_bw_data('yelp_full')
#create_bw_data('amazon_bi')
#create_bw_data('amazon_full')

#create_bw_data('eRisk_anx')


#train_lm('data/nlp_clas/imdb', 'data/nlp_clas/wikitext-103_2', lm_id='imdb_FT', backwards=False)

#train_lm('data/nlp_clas/imdb', 'data/nlp_clas/wikitext-103_2',0, lm_id='imdb_FT', backwards=True)

#train_lm('data/nlp_clas/yelp_bi/yelp_bi_lm', 'data/nlp_clas/wikitext-103_2', lm_id='yelp_FT', backwards=False)

#train_lm('data/nlp_clas/yelp_bi/yelp_bi_lm', 'data/nlp_clas/wikitext-103_2',1, lm_id='yelp_FT', backwards=True)

#train_lm('data/nlp_clas/yelp_full/yelp_full_lm', 'data/nlp_clas/wikitext-103_2', lm_id='yelpFull_FT', backwards=False)

#train_lm('data/nlp_clas/yelp_full/yelp_full_lm', 'data/nlp_clas/wikitext-103_2',2, lm_id='yelpFull_FT', backwards=True)

#train_lm('data/nlp_clas/eRisk_dep/eRisk_dep_lm', 'data/nlp_clas/wikitext-103_2', 2,  lm_id='eRisk_dep4_FT', backwards=False)

#train_lm('data/nlp_clas/amazon_bi/amazon_bi_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='amazon_bi_FT', backwards=False,bs=32)

#train_lm('data/nlp_clas/amazon_bi/amazon_bi_lm', 'data/nlp_clas/wikitext-103_2', 3, lm_id='amazon_bi_FT', backwards=True,bs=32)

#train_lm('data/nlp_clas/amazon_full/amazon_full_lm', 'data/nlp_clas/wikitext-103_2', 1, lm_id='amazon_full_FT', backwards=False,bs=32)

#train_lm('data/nlp_clas/amazon_full/amazon_full_lm', 'data/nlp_clas/wikitext-103_2', 0, lm_id='amazon_full_FT', backwards=True, bs=32)

#train_lm('data/nlp_clas/eRisk_anx/eRisk_anx_lm', 'data/nlp_clas/wikitext-103_2', 2,  lm_id='eRisk_anx_FT2', backwards=False)

train_lm('data/nlp_clas/eRisk_anx/eRisk_anx_lm', 'data/nlp_clas/wikitext-103_2', 2,  lm_id='eRisk_anx_FT2', backwards=True)

#train_lm('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_lm', 'data/nlp_clas/wikitext-103_2', 2,  lm_id='eRisk_dep_wr_FT', backwards=False)

#train_lm('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_lm', 'data/nlp_clas/wikitext-103_2', 1,  lm_id='eRisk_anx_wr_FT', backwards=False)
