
from AWDEncClas.predict_with_classifier import predict_text
from attention_visualization import *
from AWDEncClas.predict_with_classifier import load_model

#def load_model(itosFileName, classFileName):



stoi, model = load_model('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl', 'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT_lgsfx_condensedX_clas_1.h5')
classes = [False, True]





df_val = pd.read_csv('data/nlp_clas/imdb/imdb_clas/test.csv', header=None)#, chunksize=chunksize)

pos_df = df_val[df_val[0] == 1]
pos_df = pos_df.reset_index(drop=True)
pos_df.insert(0, 'cls', '--')
pos_df.insert(0, 'att', 0)
pos_df['att'] = pos_df['att'].astype(object)

neg_df = df_val[df_val[0] == 0]
neg_df = neg_df.reset_index(drop=True)
neg_df.insert(0, 'cls', '--')
neg_df.insert(0, 'att', 0)
neg_df['att'] = neg_df['att'].astype(object)

for index, row in pos_df.iterrows():

    #if index == 10:
    #    print("BREAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK")
    #    break
    truth = row[0]
    text = row[1]
    #cls, att_sc = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl',
    #                             'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT__NoCont2_clas_1.h5', text)

    scores, att_scores = predict_text(stoi, model, text)
    print('POS\t\t' + str(index) + '\t\t' + str(scores))
    cls, att_sc = classes[np.argmax(scores)], np.exp(att_scores.data.numpy())

    pos_df.loc[index, 'cls'] = cls
    pos_df.at[index, 'att'] = list(att_sc[:,0,0])
    pos_df.loc[index, 0] = truth
    pos_df.loc[index, 1] = text

print("Saving POS......")
pos_df.to_pickle("POS_val_DF.pkl")


for index, row in neg_df.iterrows():

    ##if index == 10:
    ##    print("BREAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK")
    ##    break
    truth = row[0]
    text = row[1]
    #cls, att_sc = predict_textIn('data/nlp_clas/imdb/imdb_clas/tmp/itos.pkl',
    #                             'data/nlp_clas/imdb/imdb_clas/models/fwd_imdb_FT_ATT__NoCont2_clas_1.h5', text)

    scores, att_scores = predict_text(stoi, model, text)
    print('NEG\t\t' + str(index) + '\t\t' + str(scores))
    cls, att_sc = classes[np.argmax(scores)], np.exp(att_scores.data.numpy())

    neg_df.loc[index, 'cls'] = cls
    neg_df.at[index, 'att'] = list(att_sc[:,0,0])
    neg_df.loc[index, 0] = truth
    neg_df.loc[index, 1] = text

print("Saving NEG......")
#neg_df.to_pickle("NEG_val_DF.pkl")

##################################################################################
################################################################################
#############################################################################

pos_dfR = pd.read_pickle("POS_val_DF.pkl")
neg_dfR = pd.read_pickle("NEG_val_DF.pkl")

TP_df = pos_dfR[pos_dfR['cls'] == True]
FN_df = pos_dfR[pos_dfR['cls'] == False]

TN_df = neg_dfR[neg_dfR['cls'] == False]
FN_df = neg_dfR[neg_dfR['cls'] == True]

colorStr = "0,255,0"
#createHTMLALL(FN_df.iloc[range(0,5),:], colorStr, 'htmlTrials/FN.html')
createHTMLALL(TP_df.iloc[range(0,15),:], colorStr, 'htmlTrials/TP.html')
#
#
##createHTMLALL(texts, weights, fileName)