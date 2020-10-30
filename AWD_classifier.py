import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


from AWDEncClas.train_clas import*



#train_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_paper', use_att=False, cl=14)

#train_clas('data/nlp_clas/imdb/imdb_clas', 1, lm_id='imdb_FT', clas_id='imdb_FT_paper', use_att=False, cl=14, backwards=True)

#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 2, lm_id='yelp_FT', clas_id='yelp_FT_3', use_att=False, cl=14)

#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 1, lm_id='yelp_FT', clas_id='yelp_FT_3', use_att=False, cl=14, backwards=True)

#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 0, lm_id='yelpFull_FT', clas_id='yelp_FT_14_2', use_att=False, cl=14)

#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 2, lm_id='yelpFull_FT', clas_id='yelp_FT_14_2', use_att=False, cl=14, backwards=True)

#train_clas_eRisk('data/nlp_clas/eRisk_dep/eRisk_dep_clas', 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4_3', use_att=False, cl=3)#, useWeightSampler=True)

#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_2', use_att=False, cl=50)

#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 0, lm_id='amazon_full_FT', clas_id='amazonFull_FT_2', use_att=False, cl=50, backwards=True)

#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_2', use_att=False, cl=50)

#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 2, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_2', use_att=False, cl=50, backwards=True)

#train_clas_eRisk('data/nlp_clas/eRisk_anx/eRisk_anx_clas', 1, lm_id='eRisk_anx_FT2', clas_id='eRisk_anx_2', use_att=False, cl=14)#, useWeightSampler=True)

train_clas('data/nlp_clas/eRisk_anx/eRisk_anx_clas', 2, lm_id='eRisk_anx_FT2', clas_id='eRisk_anx_3', use_att=False, cl=14, backwards=True)#, useWeightSampler=True)

#train_clas('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class', 2, lm_id='eRisk_dep_wr_FT', clas_id='eRisk_dep_wr_FT_2', use_att=False, cl=14, use_clr=False)#, useWeightSampler=True)

#train_clas('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class', 1, lm_id='eRisk_anx_wr_FT', clas_id='eRisk_anx_wr_FT', use_att=False, cl=14, use_clr=False, bs=64)#, useWeightSampler=True)

#train_clas('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class', 2, lm_id='eRisk_anx_wr_FT', clas_id='eRisk_anx_wr_FT_ACC', use_att=False, cl=50, use_clr=False, bs=64, useWeightSampler=True)

#train_clas('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class', 1, lm_id='eRisk_anx_FT2', clas_id='eRisk_anx_wr_FT_ACC', use_att=False, cl=50, use_clr=False, bs=64)#, useWeightSampler=True)