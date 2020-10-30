import numpy as np
#import fire
from AWDLSTM.create_toks_2 import FLD
import pickle


def _partition_cols_old(a,idxs):
    i=0
    for idx in idxs:
        yield a[i:i+idx]
        i+=idx
    yield a[i:]

def _partition_cols(a,idxs):
    i=0
    for idx in idxs:
        yield a[i:idx]
        i=idx
    yield a[i:]



def partition_cols(a,idxs): return list(_partition_cols_old(a,idxs))


def reverse_flds(t, fld_id):
    t = np.array(t)
    idxs = np.nonzero(t==fld_id)[0]
    parts = partition_cols(t,idxs)[1:]
    reversed = np.concatenate([np.concatenate([o[:2],o[:1:-1]]) for o in parts[::-1]])
    return reversed


def create_bw_data(prefix, joined=False):
    print(f'prefix {prefix}; joined {joined}')
    PATH=f'data/nlp_clas/{prefix}/{prefix}_lm/'
    joined_id = 'lm_' if joined else ''

    fwd_trn_path = f'{PATH}tmp/trn_{joined_id}ids.npy'
    fwd_val_path = f'{PATH}tmp/val_{joined_id}ids.npy'

    bwd_trn_path = f'{PATH}tmp/trn_{joined_id}ids_bwd.npy'
    bwd_val_path = f'{PATH}tmp/val_{joined_id}ids_bwd.npy'

    fwd_trn = np.load(fwd_trn_path)
    fwd_val = np.load(fwd_val_path)
    itos = pickle.load(open(f'{PATH}tmp/itos.pkl', 'rb'))
    stoi = {s: i for i, s in enumerate(itos)}
    fld_id = stoi[FLD]

    bwd_trn = np.array([reverse_flds(o, fld_id) for o in fwd_trn])
    bwd_val = np.array([reverse_flds(o, fld_id) for o in fwd_val])

    np.save(bwd_trn_path, bwd_trn)
    np.save(bwd_val_path, bwd_val)



def create_bw_data_class(PATH, joined=False):
    #print(f'prefix {prefix}; joined {joined}')
    #PATH=f'data/nlp_clas/{prefix}/{prefix}_lm/'
    joined_id = 'lm_' if joined else ''

    fwd_trn_path = f'{PATH}/tmp/trn_{joined_id}ids.npy'
    fwd_val_path = f'{PATH}/tmp/val_{joined_id}ids.npy'

    bwd_trn_path = f'{PATH}/tmp/trn_{joined_id}ids_bwd.npy'
    bwd_val_path = f'{PATH}/tmp/val_{joined_id}ids_bwd.npy'

    fwd_trn = np.load(fwd_trn_path)
    fwd_val = np.load(fwd_val_path)
    itos = pickle.load(open(f'{PATH}/tmp/itos.pkl', 'rb'))
    stoi = {s: i for i, s in enumerate(itos)}
    fld_id = stoi[FLD]

    bwd_trn = np.array([reverse_flds(o, fld_id) for o in fwd_trn])
    bwd_val = np.array([reverse_flds(o, fld_id) for o in fwd_val])

    np.save(bwd_trn_path, bwd_trn)
    np.save(bwd_val_path, bwd_val)



def create_bw_data_test(PATH, itosPath, joined=False):
    #print(f'prefix {prefix}; joined {joined}')
    #PATH=f'data/nlp_clas/{prefix}/{prefix}_lm/'
    joined_id = 'lm_' if joined else ''


    fwd_val_path = f'{PATH}/val_{joined_id}ids.npy'


    bwd_val_path = f'{PATH}/val_{joined_id}ids_bwd.npy'


    fwd_val = np.load(fwd_val_path)
    itos = pickle.load(open(itosPath, 'rb'))
    stoi = {s: i for i, s in enumerate(itos)}
    fld_id = stoi[FLD]


    bwd_val = np.array([reverse_flds(o, fld_id) for o in fwd_val])


    np.save(bwd_val_path, bwd_val)


if __name__ == '__main__': fire.Fire(create_bw_data)