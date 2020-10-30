import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'



from AWDEncClas.train_clas import*



#train_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_lgsfx_condensedX_paper',use_att=True, cl=14)

#train_clas('data/nlp_clas/imdb/imdb_clas', 2, lm_id='imdb_FT', clas_id='imdb_FT_ATT_ML',use_att=True, cl=14)

#train_clas('data/nlp_clas/imdb/imdb_clas', 1, lm_id='imdb_FT', clas_id='imdb_FT_ATT_ML',use_att=True, cl=14, backwards=True)

#train_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_v2',use_att=True, cl=14)

#train_clas('data/nlp_clas/imdb/imdb_clas', 1, lm_id='imdb_FT', clas_id='imdb_FT_ATT_lgsfx_condensedX_paper',use_att=True, cl=14, backwards=True)

#train_clas('data/nlp_clas/imdb/imdb_clas', 0, lm_id='imdb_FT', clas_id='imdb_FT_ATT_4rmScr',use_att=True, cl=14, from_scratch=True)
#train_clas('data/nlp_clas/imdb/imdb_clas', 1, lm_id='imdb_FT', clas_id='imdb_FT_ATT_4rmScr',use_att=True, cl=14, backwards=True, from_scratch=True)

#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 1, lm_id='yelp_FT', clas_id='yelp_FT_ATT2',use_att=True, cl=14, startat=1)

#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 1, lm_id='yelp_FT', clas_id='yelp_FT_ATT',use_att=True, cl=14, backwards=True)

#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 0, lm_id='yelp_FT', clas_id='yelp_FT_ATT2_4rmScr',use_att=True, cl=14, from_scratch=True)
#train_clas('data/nlp_clas/yelp_bi/yelp_bi_clas', 1, lm_id='yelp_FT', clas_id='yelp_FT_ATT2_4rmScr',use_att=True, cl=14, backwards=True, from_scratch=True)


#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 0, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14',use_att=True, cl=14)

#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 2, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14',use_att=True, cl=14, backwards=True)

#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 0, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14_4rmScr',use_att=True, cl=14, from_scratch=True)
#train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 1, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14_4rmScr',use_att=True, cl=14, backwards=True, from_scratch=True)


# [Test Debug Attention] train_clas('data/nlp_clas/yelp_full/yelp_full_clas', 1, lm_id='yelpFull_FT', clas_id='TEST',use_att=True, cl=14, bs=4)

#train_clas('data/nlp_clas/eRisk_dep/eRisk_dep_clas', 2, lm_id='eRisk_dep4_FT', clas_id='eRisk_dep4_ATT', use_att=True, cl=14)#, useWeightSampler=True)

#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT', use_att=True, cl=50)

#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT', use_att=True, cl=50, backwards=True)

#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 2, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT_2', use_att=True, cl=50, use_clr=False)


#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 2, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT_4rmScr', use_att=True, cl=14, from_scratch=True)
#train_clas('data/nlp_clas/amazon_full/amazon_full_clas', 1, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT_4rmScr', use_att=True, cl=14, backwards=True, from_scratch=True)


#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT', use_att=True, cl=14)

#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT', use_att=True, cl=50, use_clr=False)

#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 3, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT', use_att=True, cl=50, backwards=True)

#train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT_4rmScr_2', use_att=True, cl=14, from_scratch=True)
train_clas('data/nlp_clas/amazon_bi/amazon_bi_clas', 1, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT_4rmScr_2', use_att=True, cl=14, backwards=True, from_scratch=True)


#train_clas('data/nlp_clas/eRisk_dep_wr/eRisk_dep_wr_class', 2, lm_id='eRisk_dep_wr_FT', clas_id='eRisk_dep_wr_FT_ATT', use_att=True, cl=14, use_clr=False)#, useWeightSampler=True)

#train_clas('data/nlp_clas/eRisk_anx_wr/eRisk_anx_wr_class', 2, lm_id='eRisk_anx_wr_FT', clas_id='eRisk_anx_wr_FT_ATT', use_att=True, cl=14, use_clr=False, bs=64)#, useWeightSampler=True)