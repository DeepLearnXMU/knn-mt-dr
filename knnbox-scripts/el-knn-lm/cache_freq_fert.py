import os
import pickle
import numpy as np

from argparse import ArgumentParser
from fairseq.data import Dictionary
from collections import Counter, defaultdict
from knnbox.common_utils import Memmap, read_config

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--datastore-path', type=str)
parser.add_argument('--cache', type=str)
parser.add_argument('--overwrite', default=False, action='store_true')
parser.add_argument('--dict-path', type=str, required=True)
parser.add_argument('--csize', type=int, default=1)
args = parser.parse_args()

def get_ngram_freq(token_indices, ngram=4, dictionary=None):
    res = Counter()
    prev = [dictionary.index('</s>')] * ngram
    for tk_ind in token_indices:
        prev = prev[-ngram:]
        for j in range(max(ngram - 1, 1), ngram + 1):
            res[tuple(prev[-j:])] += 1

        prev.append(tk_ind)
        
    return Counter({
        key: np.log(val + 1).astype('float32')
        for key, val in res.items()
    })

def get_ngram_fert(token_indices, ngram=4, dictionary=None):
    res = defaultdict(set)
    prev = [dictionary.index('</s>')] * ngram
    for tk_ind in token_indices:
        prev = prev[-ngram:]
        for j in range(max(ngram - 1, 1), ngram + 1):
            res[tuple(prev[-j:])].add(tk_ind)
        
        prev.append(tk_ind)
    
    return Counter({
        key: np.log(len(res[key]) + 1).astype('float32')
        for key in res
    })

freq_cache = f'{args.cache}/freq_train_cnt.pickle'
fert_cache = f'{args.cache}/fert_train_cnt.pickle'
dictionary = Dictionary.load(args.dict_path)

config = read_config(args.datastore_path)["data_infos"]
vals = Memmap(f'{args.datastore_path}/vals.npy', shape=config["vals"]["shape"], dtype=config["vals"]["dtype"], mode="r+")
token_indices = vals.data.tolist()

if not args.overwrite and os.path.isfile(freq_cache):
    print('skip freq cache fiiles since they exist')
else:
    print('compute freq statistics')
    freq_cnt = get_ngram_freq(token_indices, ngram=args.csize, dictionary=dictionary)
    with open(freq_cache, 'wb') as f:
        pickle.dump(freq_cnt, f)

if not args.overwrite and os.path.isfile(fert_cache):
    print('skip fert cache fiiles since they exist')
else:
    print('compute fert statistics')
    fert_cnt = get_ngram_fert(token_indices, ngram=args.csize, dictionary=dictionary)
    with open(fert_cache, 'wb') as f:
        pickle.dump(fert_cnt, f)