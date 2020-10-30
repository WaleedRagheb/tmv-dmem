import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from AWDEncClas.eval_clas import *



#print("Evaluation of classifier without Attention:\n")
#eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT')

#eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_paper')
#print("Evaluation of classifier WITH Attention:\n")
#eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT__NoCont2',attention=True)


#eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_lgsfx_condensedX',attention=True)

#eval_clas_AttentionScores('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_lgsfx_condensedX',attention=True)


#eval_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_lgsfx_condensedX_paper',attention=True)

#print("Evaluation of classifier without Attention:\n")
#eval_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 3, lm_id='yelp_FT', clas_id='yelp_FT', attention=False)
#print("Evaluation of classifier WITH Attention:\n")
#eval_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 2, lm_id='yelp_FT', clas_id='yelp_FT_ATT_lgsfx', attention=True)

#eval_clas_AttentionScores('data/nlp_clas/yelp_bi/yelp_bi_clas', 3, lm_id='yelp_FT', clas_id='yelp_FT_ATT_lgsfx', attention=True)


#print("Evaluation of classifier without Attention:\n")
#eval_clas('data/nlp_clas/yelp_full/yelp_full_clas', 1, lm_id='yelpFull_FT', clas_id='yelp_FT', attention=False)
#print("Evaluation of classifier WITH Attention:\n")
#eval_clas('data/nlp_clas/yelp_full/yelp_full_clas', 1, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14', attention=True)
#

#eval_clas('data/nlp_clas/eRisk_dep/eRisk_dep_clas', 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4_2_BEST', attention=False)



#print("Evaluation of classifier without Attention:\n")
#eval_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_2', attention=False)
#print("Evaluation of classifier WITH Attention:\n")
#eval_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT_BEST', attention=True)


#print("Evaluation of classifier without Attention:\n")
#eval_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT', attention=False)
#print("Evaluation of classifier WITH Attention:\n")
#eval_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT_BEST', attention=True)

eval_clas('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class', 2, lm_id='eRisk_dep_wr_FT', clas_id='eRisk_dep_wr_FT_ATT', attention=True)