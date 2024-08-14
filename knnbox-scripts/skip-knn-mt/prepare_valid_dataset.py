import os
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from argparse import ArgumentParser
from fairseq import utils
from fairseq.data import Dictionary

from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.common_utils import Memmap, read_config

knn_params = {
    'koran': [0.8, 100],
    'it': [0.7, 10],
    'medical': [0.8, 10],
    'law': [0.8, 10],
    'subtitles': [0.7, 10],
}

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model-path', type=str)
parser.add_argument('--datastore-path', type=str)
parser.add_argument('--save-path', type=str)
parser.add_argument('--knn-k', type=int)
args = parser.parse_args()
args.knn_lambda, args.knn_tempe = knn_params[args.dataset]

vocab = Dictionary.load(f'{args.model_path}/fairseq-vocab.txt')
datastore = Datastore.load(args.datastore_path, load_list=["keys", "vals"])
datastore.load_faiss_index('keys', move_to_gpu=True)
config = read_config(f'{args.datastore_path}/valid')["data_infos"]
keys_mmp = Memmap(f'{args.datastore_path}/valid/keys.npy', shape=config["keys"]["shape"], dtype=config["keys"]["dtype"], mode="r+")
vals_mmp = Memmap(f'{args.datastore_path}/valid/vals.npy', shape=config["vals"]["shape"], dtype=config["vals"]["dtype"], mode="r+")
attns_mmp = Memmap(f'{args.datastore_path}/valid/attns.npy', shape=config["attns"]["shape"], dtype=config["attns"]["dtype"], mode="r+")


keys = torch.from_numpy(keys_mmp.data[:]).type(torch.float32)
vals = torch.from_numpy(vals_mmp.data[:])
attns = torch.from_numpy(attns_mmp.data[:])

print('finished loading datastore')

output_projection_ckp = torch.load(f'{args.model_path}/output_projection.pt')
output_projection = nn.Linear(1024, len(vocab), bias=False)
output_projection.load_state_dict(output_projection_ckp)
output_projection = output_projection.cuda()
print('finished loading model')

retriever = Retriever(datastore=datastore, k=args.knn_k)

query = []
nmt_prob = []
knn_approximate_dist = []
knn_tgt = []
knn_top_key = []
keys_index = torch.arange(len(keys))
bsz = 1024
with torch.no_grad():
    for index in tqdm(keys_index.split(bsz, 0), total=len(keys_index) // bsz, desc='processing'):
        x = keys[index].cuda()
        target = vals[index].cuda()
        nmt_probs = torch.softmax(output_projection(x), dim=-1)
        res = retriever.retrieve(x.unsqueeze(1), return_list=["vals", "distances", "keys"], remove_first=False)

        knn_key = res["keys"].squeeze()
        knn_dist = res["distances"].squeeze()
        knn_val = res["vals"].squeeze()

        knn_probs = torch.zeros_like(nmt_probs)
        knn_spare_distribution = torch.softmax(- knn_dist / args.knn_tempe, dim=-1)
        knn_probs.scatter_add_(dim=-1, index=knn_val, src=knn_spare_distribution)

        query.append(x.cpu())
        knn_approximate_dist.append(knn_dist.cpu())
        knn_tgt.append(knn_val.cpu())
        knn_top_key.append(knn_key.cpu())


query = torch.cat(query, dim=0)
knn_approximate_dist = torch.cat(knn_approximate_dist, dim=0)
knn_tgt = torch.cat(knn_tgt, dim=0)
knn_top_key = torch.cat(knn_top_key, dim=0).float()

torch.save(query, f'{args.save_path}/query.pt')
torch.save(vals.squeeze(), f'{args.save_path}/target.pt')


torch.save(attns, f'{args.save_path}/attn.pt')



torch.save(knn_approximate_dist, f'{args.save_path}/knn_dist.pt')
torch.save(knn_tgt, f'{args.save_path}/knn_tgt.pt')
torch.save(knn_top_key, f'{args.save_path}/knn_key.pt')

print('finished prepare dataset')
