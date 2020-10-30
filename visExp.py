from AWDEncClas.predict_with_classifier import *
from attention_visualization import *
from AWDEncClas.eval_clas import attentionLexicBased


key=""
text = "I love the idea of this place but I bought a groupon and you have to sign in on line within 30 days or it won't let you and they never answer the phone or return phone calls or email and when you go by no one is there I don't know how they keep running specials I suggest don't by a group on and the instructors aren't very pleasant to be around good luck I had to contact groupon to get my money back to purchase another if this happens to you group on is wonderful they will do what it is you want they will even contact tough lotus if you want."

#cls, att_sc, outStr = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl','data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_lgsfx_condensedX_clas_1.h5',text)

cls, att_sc, outStr = predict_textIn('data/nlp_clas/yelp_bi/yelp_bi_clas/tmp/itos.pkl', 'data/nlp_clas/yelp_bi/yelp_bi_clas/models/fwd_yelp_FT_ATT_lgsfx_clas_1.h5',text)

att_lex = attentionLexicBased(outStr)
fileName = 'Figs/RealQuiz/' + key + '.png'
fileName_lex = 'Figs/RealQuiz/' + key + '_lex.png'
fileName_Mix = 'Figs/RealQuiz/' + key + '_Mix.png'

