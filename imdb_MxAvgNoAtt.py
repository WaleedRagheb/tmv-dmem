import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from fastai.learner import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle
import spacy


PATH='/home/waleed/data/imdb/imdb2/aclImdb/'

TRN_PATH = 'train/all/'
VAL_PATH = 'test/all/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

#review = !type {TRN}{trn_files[6]}
#os.system("type {TRN}{trn_files[6]")
spacy_tok = spacy.load('en')


em_sz = 200  # size of each embedding vector
nh = 500     # number of hidden activations per layer
nl = 3       # number of layers
bs=64; bptt=70
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

TEXT = data.Field(lower=True, tokenize="spacy")

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)
pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))

len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)



###########################[ Encoder ]########################################

learner = md.get_model(opt_fn, em_sz, nh, nl,
               dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3

learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)

learner.save_encoder('adam1_enc')
learner.load_encoder('adam1_enc')
learner.fit(3e-3, 1, wds=1e-6, cycle_len=10)

learner.save_encoder('adam3_10_enc')
learner.load_encoder('adam3_10_enc')
math.exp(4.165)
pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))



m=learner.model
ss=""". So, it wasn't quite was I was expecting, but I really liked it anyway! The best"""
s = [TEXT.preprocess(ss)]
t=TEXT.numericalize(s)
' '.join(s[0])

# Set batch size to 1
m[0].bs=1
# Turn off dropout
m.eval()
# Reset hidden state
m.reset()
# Get predictions from model
res,*_ = m(t)
# Put the batch size back to what it was
m[0].bs=bs

nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]

import warnings
warnings.filterwarnings("ignore")

print(ss,"\n")
for i in range(50):
    n=res[-1].topk(2)[1]
    n = n[1] if n.data[0]==0 else n[0]
    #print(i)
    print(TEXT.vocab.itos[n], end=' ')
    res,*_ = m(n[0].unsqueeze(0).unsqueeze(0))
print('...')

#####################################[ Classifier Training ]########################################

import warnings
warnings.filterwarnings("ignore")

#########################################################

TEXT = pickle.load(open(f'{PATH}/models/TEXT.pkl','rb'))
IMDB_LABEL = data.Field(sequential=False)
splits = torchtext.datasets.IMDB.splits(TEXT, IMDB_LABEL, f'{PATH}')
t = splits[0].examples[0]
print(t.label)
print(' '.join(t.text[:16]))

md2 = TextData.from_splits(PATH, splits, bs)
m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl,
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.load_encoder(f'adam3_10_enc')
m3.clip=25.
lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])
m3.freeze_to(-1)
m3.fit(lrs/2, 1, metrics=[accuracy])
m3.unfreeze()
m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1)
#m3.fit(lrs, 7, metrics=[accuracy], cycle_len=2, cycle_save_name='imdb2')

accuracy_np(*m3.predict_with_targs())




