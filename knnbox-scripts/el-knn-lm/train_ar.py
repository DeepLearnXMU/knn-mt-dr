import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import numpy as np

from argparse import ArgumentParser
from collections import Counter, OrderedDict
from knnbox.modules import MLPMOE
from knnbox.token_feature_dataset import TokenFeatureDataset

def validate(valid_dataloader, model, args):
    model.eval()
    model.epoch_update()
    running_loss = 0.
    nsamples = 0
    prediction_dict = {}
    for i, sample in enumerate(valid_dataloader, 0):
        inputs, nmt_scores, knn_scores= sample['feature'], sample['nmt_prob'], sample['knn_prob']
        log_weight = model(inputs)


        scores_cat = torch.stack([nmt_scores + 1e-9, knn_scores + 1e-9], dim=-1).log()
        cross_entropy = log_weight + scores_cat

        cross_entropy = - torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()
        ent_loss = loss

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        bsz = next(iter(inputs.values())).size(0)

        running_loss += loss.item() * bsz
        nsamples += bsz

    val_loss = running_loss / nsamples
    print(f"val loss: {val_loss:.3f}")
    # print(f"val loss: {val_loss:.3f}, ppl: {np.exp(val_loss):.3f}")
    return val_loss

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l1', type=float, default=0.)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--ngram', type=int, default=0)
parser.add_argument('--activation', type=str, default='relu', choices=['linear', 'relu'])
parser.add_argument('--hidden-units', type=int, default=32)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--cache-path', type=str)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

fert_data = read_pickle(f'{args.cache_path}/fert_cache.pickle').float()
freq_data = read_pickle(f'{args.cache_path}/freq_cache.pickle').float()
knn_prob = read_pickle(f'{args.cache_path}/knn_prob.pickle')
nmt_prob = read_pickle(f'{args.cache_path}/nmt_prob.pickle')
nmt_max = read_pickle(f'{args.cache_path}/nmt_max.pickle')
nmt_entropy = read_pickle(f'{args.cache_path}/nmt_entropy.pickle')
ctext = read_pickle(f'{args.cache_path}/ctxt.pickle')
print('complete reading pickle files')

indexes = np.arange(len(nmt_prob))
train_size = 0.8
np.random.shuffle(indexes)
boundary = int(len(indexes) * train_size)
train_indexes = torch.from_numpy(indexes[:boundary])
valid_indexes = torch.from_numpy(indexes[boundary:])

train_dataset = TokenFeatureDataset(
    fert_data=fert_data[train_indexes],
    freq_data=freq_data[train_indexes],
    knn_prob=knn_prob[train_indexes],
    nmt_prob=nmt_prob[train_indexes],
    nmt_max=nmt_max[train_indexes],
    nmt_entropy=nmt_entropy[train_indexes],
    ctext=ctext[train_indexes],
    ngram=args.ngram,
)

valid_dataset = TokenFeatureDataset(
    fert_data=fert_data[valid_indexes],
    freq_data=freq_data[valid_indexes],
    knn_prob=knn_prob[valid_indexes],
    nmt_prob=nmt_prob[valid_indexes],
    nmt_max=nmt_max[valid_indexes],
    nmt_entropy=nmt_entropy[valid_indexes],
    ctext=ctext[valid_indexes],
    ngram=args.ngram,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=None,
    collate_fn=train_dataset.collater
)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=None,
    collate_fn=valid_dataset.collater
)

print('complete constructing dataloader')

feature_set = ['ctxt', 'freq', 'nmt_entropy', 'nmt_max', 'fert']
feature_size = OrderedDict({key: train_dataset.get_nfeature(key) for key in feature_set})

model = MLPMOE(
    feature_size=feature_size,
    hidden_units=args.hidden_units,
    nlayers=args.nlayers,
    dropout=args.dropout,
    activation=args.activation,
).cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()

# for i, (name, params) in enumerate(model.named_parameters()):
#     if i == 2:
#         break
#     print(f'{name}: {params.mean():.6f} - {params.var():.6f}')

nepochs = 10
best_loss = 1e5
best_half_cut_ppl = 1e5
for epoch in range(nepochs):
    running_loss = 0.
    nsamples = 0

    model.epoch_update()

    for i, sample in enumerate(train_dataloader):

        inputs, nmt_scores, knn_scores = sample['feature'], sample['nmt_prob'], sample['knn_prob']

        optimizer.zero_grad()
        log_weight = model(inputs)


        scores_cat = torch.stack([nmt_scores + 1e-9, knn_scores + 1e-9], dim=-1).log()
        cross_entropy = log_weight + scores_cat
        cross_entropy = - torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()
        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)
        
        loss.backward()
        optimizer.step()

        bsz = next(iter(inputs.values())).size(0)
        running_loss += loss.item() * bsz
        nsamples += bsz

        if (i + 1) % 100 == 0:
            report_loss = running_loss / nsamples
            print(f'epoch: {epoch}, step: {i + 1}, training loss: {report_loss:.3f}')
            # print(f'epoch: {epoch}, step: {i + 1}, training loss: {report_loss:.3f}, ppl: {np.exp(report_loss):.3f}')

            # for i, (name, params) in enumerate(model.named_parameters()):
            #     if i == 2:
            #         break
            #     print(f'{name}: {params.mean():.6f} - {params.var():.6f}')

    val_loss = validate(valid_dataloader, model, args)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'{args.cache_path}/checkpoint_best.pt')
    model.train()
