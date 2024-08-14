import os
import pickle

import torch
import faiss
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from fairseq import metrics, utils

from argparse import ArgumentParser
from fairseq.data import Dictionary

from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
from knnbox.common_utils import Memmap, read_config

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model-path', type=str)
parser.add_argument('--datastore-path', type=str)
parser.add_argument('--cache', type=str)
parser.add_argument('--knn-lambda', type=float)
parser.add_argument('--knn-temperature', type=int)
parser.add_argument('--knn-k', type=int)
parser.add_argument('--dict-path', type=str, required=True)
parser.add_argument('--csize', type=int)
parser.add_argument('--subset', type=str, default='valid')
args = parser.parse_args()

ngram = args.csize
subset = args.subset
dictionary = Dictionary.load(args.dict_path)
datastore = Datastore.load(args.datastore_path, load_list=["keys", "vals"])
datastore.load_faiss_index('keys', move_to_gpu=True)
config = read_config(f'{args.datastore_path}/{subset}')["data_infos"]
keys_mmp = Memmap(f'{args.datastore_path}/{subset}/keys.npy', shape=config["keys"]["shape"], dtype=config["keys"]["dtype"], mode="r+")
vals_mmp = Memmap(f'{args.datastore_path}/{subset}/vals.npy', shape=config["vals"]["shape"], dtype=config["vals"]["dtype"], mode="r+")
#sentence_ids_mmp = Memmap(f'{args.datastore_path}/{subset}/sentence_ids.npy', shape=config["sentence_ids"]["shape"], dtype=config["sentence_ids"]["dtype"], mode="r+")
# positions_tgt_mmp = Memmap(f'{args.datastore_path}/{subset}/positions_tgt.npy', shape=config["positions_tgt"]["shape"], dtype=config["positions_tgt"]["dtype"], mode="r+")

keys = torch.from_numpy(keys_mmp.data[:]).type(torch.float32).cuda()
vals = torch.from_numpy(vals_mmp.data[:]).cuda()
#sentence_ids = torch.from_numpy(sentence_ids_mmp.data[:]).cuda()
# positions_tgt = torch.from_numpy(positions_tgt_mmp.data[:])


embed_weight = torch.load(args.model_path)['model']['decoder.embed_tokens.weight']
output_projection = nn.Linear(
    embed_weight.shape[1],
    embed_weight.shape[0],
    bias=False,
)
output_projection.weight.data = embed_weight
output_projection = output_projection.cuda()

combiner = Combiner(lambda_=args.knn_lambda, temperature=args.knn_temperature, probability_dim=len(dictionary))
retriever = Retriever(datastore=datastore, k=args.knn_k)

with open(f'{args.cache}/freq_train_cnt.pickle', 'rb') as f:
    freq_cnt = pickle.load(f)

with open(f'{args.cache}/fert_train_cnt.pickle', 'rb') as f:
    fert_cnt = pickle.load(f)

nmt_max_list = []
nmt_entropy_list = []

nmt_prob_list = []
knn_prob_list = []

fert_list = []
freq_list = []

if subset == 'valid':
    ctxt_cache = f'{args.cache}/ctxt.pickle'
    nmt_max_cache = f'{args.cache}/nmt_max.pickle'
    nmt_entropy_cache = f'{args.cache}/nmt_entropy.pickle'
    nmt_prob_cache = f'{args.cache}/nmt_prob.pickle'
    knn_prob_cache = f'{args.cache}/knn_prob.pickle'
    freq_cache = f'{args.cache}/freq_cache.pickle'
    fert_cache = f'{args.cache}/fert_cache.pickle'
    # positions_tgt_cache = f'{args.cache}/positions_tgt_cache.pickle'
    #sentence_ids_cache = f'{args.cache}/sentence_ids.pickle'
else:
    ctxt_cache = f'{args.cache}/{subset}.ctxt.pickle'
    nmt_max_cache = f'{args.cache}/{subset}.nmt_max.pickle'
    nmt_entropy_cache = f'{args.cache}/{subset}.nmt_entropy.pickle'
    nmt_prob_cache = f'{args.cache}/{subset}.nmt_prob.pickle'
    knn_prob_cache = f'{args.cache}/{subset}.knn_prob.pickle'
    freq_cache = f'{args.cache}/{subset}.freq_cache.pickle'
    fert_cache = f'{args.cache}/{subset}.fert_cache.pickle'
    # positions_tgt_cache = f'{args.cache}/{subset}.positions_tgt_cache.pickle'
    #sentence_ids_cache = f'{args.cache}/{subset}.sentence_ids.pickle'

prev = [dictionary.index('</s>')] * ngram
for x, y in tqdm(zip(keys.split(1, 0), vals.split(1, 0)), total=len(keys)):

    x = x.unsqueeze(1)
    prev = prev[-ngram:]
    
    with torch.no_grad():
        res = retriever.retrieve(x, return_list=["vals", "distances"], remove_first=False)
        nmt_prob = utils.softmax(output_projection(x), dim=-1)

        knn_dists = res["distances"]
        tgt_index = res["vals"]

        knn_prob = combiner.get_knn_prob(vals=tgt_index, distances=knn_dists, device=nmt_prob.device)
        # combined_prob, _ = combiner.get_combined_prob(knn_prob, nmt_prob, log_probs=False)

        nmt_max = nmt_prob.max(dim=-1).values.squeeze()
        nmt_entropy = - (nmt_prob * nmt_prob.log()).sum(dim=-1).squeeze()

        nmt_target_prob = nmt_prob.squeeze()[y].squeeze()
        knn_target_prob = knn_prob.squeeze()[y].squeeze()

        nmt_max_list.append(nmt_max)
        nmt_entropy_list.append(nmt_entropy)
        nmt_prob_list.append(nmt_target_prob)
        knn_prob_list.append(knn_target_prob)

        freq_list.append([
            freq_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)
        ])
        fert_list.append([
            fert_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)
        ])
    
    prev.append(y.squeeze().item())

nmt_max_list = torch.stack(nmt_max_list, dim=-1).cpu()
nmt_entropy_list = torch.stack(nmt_entropy_list, dim=-1).cpu()
nmt_prob_list = torch.stack(nmt_prob_list, dim=-1).cpu()
knn_prob_list = torch.stack(knn_prob_list, dim=-1).cpu()

freq_list = torch.tensor(freq_list).squeeze()
fert_list = torch.tensor(fert_list).squeeze()

def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

print('save statistics results')
save_pickle(nmt_max_list, nmt_max_cache)
save_pickle(nmt_entropy_list, nmt_entropy_cache)
save_pickle(nmt_prob_list, nmt_prob_cache)
save_pickle(knn_prob_list, knn_prob_cache)
save_pickle(freq_list, freq_cache)
save_pickle(fert_list, fert_cache)
save_pickle(keys, ctxt_cache)
#save_pickle(sentence_ids, sentence_ids_cache)
# save_pickle(positions_tgt, positions_tgt_cache)
