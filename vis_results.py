from AWDEncClas.predict_with_classifier import *
from attention_visualization import *
from AWDEncClas.eval_clas import eval_clas_AttentionScores


#eval_clas_AttentionScores('data/nlp_clas/yelp_full/yelp_full_clas', 0, lm_id='yelpFull_FT', clas_id='yelp_FT_ATT_14', attention=True)

#eval_clas_AttentionScores('data/nlp_clas/amazon_full/amazon_full_clas', 0, lm_id='amazon_full_FT', clas_id='amazonFull_FT_ATT_BEST', attention=True)

#eval_clas_AttentionScores('data/nlp_clas/amazon_bi/amazon_bi_clas', 0, lm_id='amazon_bi_FT', clas_id='amazonBi_FT_ATT', attention=True)

####################################################
####################################################
#text = "Van Sant copies Hitchcock's masterpiece shot for shot including some modern facets: a walkman, and nudity from Anne Heche. Unless you have a strong desire to see Ms. Heche naked there is absolutely NO reason to see this film instead of the original. Hitchcock's masterpiece is much better and Van Sant fails to realize that in hiding the nudity and the gore, the original shower scene is all the more terrifying. Ask Janet Leigh about that one. The acting is also much flatter and the technical aspects much less impressive."
text = "This movie is good but not amazing"
#
cls, att_sc, outStr = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl', 'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_ML_clas_1.h5', text)

l1_att = att_sc[0].data.numpy()
l2_att = att_sc[1].data.numpy()
l3_att = att_sc[2].data.numpy()


#createHTMLALL_OneText([outStr], [list(att_sc[:,0,0])], 'htmlTrials/survey/Yelp7.html')
createHTMLALL_OneText([outStr], [list(l1_att[:,0,0])], 'htmlTrials/survey/exp_1_l1.html')
createHTMLALL_OneText([outStr], [list(l2_att[:,0,0])], 'htmlTrials/survey/exp_1_l2.html')
createHTMLALL_OneText([outStr], [list(l3_att[:,0,0])], 'htmlTrials/survey/exp_1_l3.html')