#import fire
from fastai.text import *
from fastai.lm_rnn import *
from sklearn.metrics import confusion_matrix

from attention_visualization import *

def eval_clas(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'lbl_val.npy').flatten()
    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)
    predictions = np.argmax(learn.predict(), axis=1)
    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls_sampled, predictions))


##########################################################################################################
#########################################################################################################
def processAllLexic(textList, attentionList,percentage):

    i, cntr =0, 0

    lexicList = []

    with open('data/NRC-Emotion-Lexicon.txt', 'r') as f:
        for line in f:
            for word in line.split():
                lexicList.append(word)

    for txt in textList:
        #print(i)
        tokVec = txt.split("~~")
        tokSize = len(tokVec)

        #TXTVec.append(texts)

        att = (attentionList[i])[4:tokSize-1]
        attVec = torch.stack(att).cpu().numpy()
        tempDic = {'tokens': tokVec[5:], 'attention': attVec}
        if(len(attVec) != len(tokVec[5:])):
            continue
        tempDF = pd.DataFrame(tempDic)

        sortedDF =tempDF.sort_values(by = 'attention',ascending=False)

        topN_DF = sortedDF.head(int(tokSize * percentage))
        word_list = topN_DF['tokens'].tolist()

        intersections = list(set(lexicList) & set(word_list))
        if len(intersections) > 0:
            cntr = cntr + 1

        #print(len(intersections))

        i = i+1

    res = cntr / len(textList)

    print('Results for :' + str(percentage) + ' :' + str(res))

    return res


def compareWithLexic(dfALL,percentage):
    #percentage = 0.05
    textAll = dfALL.iloc[range(0, len(dfALL)), 0].as_matrix()
    att_All = dfALL.iloc[range(0, len(dfALL)), 3].as_matrix()
    res = processAllLexic(textAll, att_All, percentage)


#########################################################################################################
#########################################################################################################

def attentionLexicBased(txt):
    lexicList = []
    with open('data/NRC-Emotion-Lexicon.txt', 'r') as f:
        for line in f:
            for word in line.split():
                lexicList.append(word)

    lexicList = list(set(lexicList))

    tok = txt.split("~~")
    att = [0 if w in lexicList else -100 for w in tok]
    return att



#########################################################################################################
#########################################################################################################

def eval_clas_AttentionScores(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'lbl_val.npy').flatten()
    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        itos_dic = {i: s for i, s in enumerate(itos)}
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 40*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)

    pred, attscr = learn.predict_withAtt()

    predictions = np.argmax(np.concatenate(pred), axis=1)
    val_sent_samp =  val_sent[list(val_samp)]
    val_sent_txt = []
    for lst in val_sent_samp:
        text = ""
        for nn in lst:
            #if(itos_dic[nn].find('~~')!=-1):
            #    print(itos_dic[nn])
            text = text + "~~" + itos_dic[nn].replace(" ", "")
        val_sent_txt.append(text.strip())

    attentionScores = []
    for ss in range(len(attscr)):
        attsz,bss,u = attscr[ss].size()
        for ii in range(bss):
            sc = list(attscr[ss][:,ii,0])
            attentionScores.append(sc)

    df_val = pd.read_csv(dir_path / 'test.csv', header=None)
    origTextList = df_val.iloc[:,1].values
    origTextList_samp = origTextList[list(val_samp)]


    dicALL = {'text': val_sent_txt, 'label': val_lbls_sampled, 'prediction': predictions, 'attention': attentionScores, 'OriginalText': origTextList_samp}
    dfALL = pd.DataFrame(data=dicALL)





    # process the attscr with val_samp with val_lbls_sampled
    # note: len(val_sent[list(val_samp)][24000])
    # don't forget to back to text

    #predictions = np.argmax(learn.predict(), axis=1)
    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')
    print(confusion_matrix(val_lbls_sampled, predictions))

    compareWithLexic(dfALL,0.05)
    compareWithLexic(dfALL, 0.1)
    compareWithLexic(dfALL, 0.2)

    #TP_df = dfALL[(dfALL['prediction'] == 1) & (dfALL['label'] == 1)]
    #FN_df = dfALL[(dfALL['prediction'] == 0) & (dfALL['label'] == 1)]
#
    #TN_df = dfALL[(dfALL['prediction'] == 0) & (dfALL['label'] == 0)]
    #FP_df = dfALL[(dfALL['prediction'] == 1) & (dfALL['label'] == 0)]
#
    #colorStr = "0,255,0"
    #createHTMLALL_att(TP_df.iloc[random.sample(range(11848), 20), :], colorStr, 'htmlTrials/yelp/TP_N.html')
    #createHTMLALL_att(TN_df.iloc[random.sample(range(11913), 20), :], colorStr, 'htmlTrials/yelp/TN_N.html')
#
    #colorStr = "255,0,0"
    #createHTMLALL_att(FP_df.iloc[random.sample(range(400), 20), :], colorStr, 'htmlTrials/yelp/FP_N.html')
    #createHTMLALL_att(FN_df.iloc[random.sample(range(400), 20), :], colorStr, 'htmlTrials/yelp/FN_N.html')




###################################################################################################################
###################################################################################################################


def eval_clas_eRisk(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'tests' / 'lbl_val.npy').flatten()


    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'tests' / 'itos.pkl', 'rb'))
        itos_dic = {i: s for i, s in enumerate(itos)}
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 40*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)

    pred = learn.predict()

    predictions = np.argmax(pred, axis=1)

    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')

    print(confusion_matrix(val_lbls_sampled, predictions))

    pos_prob = [softmax(p)[1] for p in pred]

    return predictions, list(val_samp), pos_prob


    #compareWithLexic(dfALL,0.05)
    #compareWithLexic(dfALL, 0.1)
    #compareWithLexic(dfALL, 0.2)

    #TP_df = dfALL[(dfALL['prediction'] == 1) & (dfALL['label'] == 1)]
    #FN_df = dfALL[(dfALL['prediction'] == 0) & (dfALL['label'] == 1)]
#
    #TN_df = dfALL[(dfALL['prediction'] == 0) & (dfALL['label'] == 0)]
    #FP_df = dfALL[(dfALL['prediction'] == 1) & (dfALL['label'] == 0)]
#
    #colorStr = "0,255,0"
    #createHTMLALL_att(TP_df.iloc[random.sample(range(11848), 20), :], colorStr, 'htmlTrials/yelp/TP_N.html')
    #createHTMLALL_att(TN_df.iloc[random.sample(range(11913), 20), :], colorStr, 'htmlTrials/yelp/TN_N.html')
#
    #colorStr = "255,0,0"
    #createHTMLALL_att(FP_df.iloc[random.sample(range(400), 20), :], colorStr, 'htmlTrials/yelp/FP_N.html')
    #createHTMLALL_att(FN_df.iloc[random.sample(range(400), 20), :], colorStr, 'htmlTrials/yelp/FN_N.html')

##################################################################################################################


def eval_clas_eRisk_final(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'dummy' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'dummy' / f'val_{IDS}.npy')
    #val_lbls = np.load(dir_path / 'tmp' / 'tests' / 'dummy' / 'lbl_val.npy').flatten()


    c=2
    val_lbls = np.zeros(len(val_sent))

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'tests' / 'itos.pkl', 'rb'))
        itos_dic = {i: s for i, s in enumerate(itos)}
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 40*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)

    pred = learn.predict()

    predictions = np.argmax(pred, axis=1)

    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')

    print(confusion_matrix(val_lbls_sampled, predictions))

    pos_prob = [softmax(p)[1] for p in pred]

    return predictions, list(val_samp), pos_prob



##################################################################################################################

def eval_clas_eRisk_final_2(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'test' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'test' / f'val_{IDS}.npy')
    #val_lbls = np.load(dir_path / 'tmp' / 'tests' / 'dummy' / 'lbl_val.npy').flatten()


    c=2
    val_lbls = np.zeros(len(val_sent))

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'tests' / 'itos.pkl', 'rb'))
        itos_dic = {i: s for i, s in enumerate(itos)}
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 40*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)

    pred = learn.predict()

    predictions = np.argmax(pred, axis=1)

    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')

    print(confusion_matrix(val_lbls_sampled, predictions))

    pos_prob = [softmax(p)[1] for p in pred]

    return predictions, list(val_samp), pos_prob



##################################################################################################################
##################################################################################################################

def eval_clas_eRisk_final_3(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'testT2' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / 'tests' / 'testT2' / f'val_{IDS}.npy')
    #val_lbls = np.load(dir_path / 'tmp' / 'tests' / 'dummy' / 'lbl_val.npy').flatten()


    c=2
    val_lbls = np.zeros(len(val_sent))

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'tests' / 'itos.pkl', 'rb'))
        itos_dic = {i: s for i, s in enumerate(itos)}
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 40*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                  layers=[em_sz*3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)

    pred = learn.predict()

    predictions = np.argmax(pred, axis=1)

    acc = (val_lbls_sampled == predictions).mean()
    print('Accuracy =', acc, 'Confusion Matrix =')

    print(confusion_matrix(val_lbls_sampled, predictions))

    pos_prob = [softmax(p)[1] for p in pred]

    return predictions, list(val_samp), pos_prob



##################################################################################################################


def eval_clas_ens(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, backwards=False,
              bpe=False, attention = False, scores=False):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; '
         f'clas_id {clas_id}; bs {bs}; backwards {backwards}; bpe {bpe}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3

    if backwards:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')
    val_lbls = np.load(dir_path / 'tmp' / 'lbl_val.npy').flatten()
    c=int(val_lbls.max())+1

    val_ds = TextDataset(val_sent, val_lbls)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    val_lbls_sampled = val_lbls[list(val_samp)]
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, None, val_dl)

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
        vs = len(itos)
    if attention:
        m = get_rnn_classifier_w(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
        #if backwards:
        #    m = get_rnn_classifier_w_ML(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
        #              layers=[em_sz*3, 50, c], drops=[0., 0.])
        #else:
        #    m = get_rnn_classifier_w(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
        #                         layers=[em_sz * 3, 50, c], drops=[0., 0.])
    else:
        m = get_rnn_classifier(bptt, 20 * 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
                                 layers=[em_sz * 3, 50, c], drops=[0., 0.])
    learn = RNN_Learner(md, TextModel(to_gpu(m)))
    learn.load(final_clas_file)
    pred = learn.predict()

    if scores:
        return pred, list(val_samp), val_lbls_sampled
    else:
        predictions = np.argmax(pred, axis=1)

        acc = (val_lbls_sampled == predictions).mean()
        print('Accuracy =', acc, 'Confusion Matrix =')

        print(confusion_matrix(val_lbls_sampled, predictions))
        return predictions, list(val_samp)



#if __name__ == '__main__': fire.Fire(eval_clas)

