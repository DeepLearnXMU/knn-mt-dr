import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import pickle
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain

from argparse import ArgumentParser
from me_dataset import MeDataset
from criterion import MeCriterion

from fairseq.data import Dictionary
from knnbox.combiner import SkipCombiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
logger = logging.getLogger('train')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--datastore-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--base-model-path', type=str)
    parser.add_argument('--knn-max-k', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-epoch', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--move-data-to-gpu', default=False, action='store_true')
    parser.add_argument('--with-adaptive', default=False, action='store_true')
    parser.add_argument('--alpha-coef', default=0., type=float)
    parser.add_argument('--alpha-mode', default='v1', type=str)
    parser.add_argument('--norm-coef', default=2., type=float)
    parser.add_argument('--balance-coef', default=1.5, type=float)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--validation-skip-percent', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=10)
    parser.add_argument('--knn-lambda', type=float, default=0.7)

    args = parser.parse_args()
    return args
    

def gather_list_and_reduce(logging_outputs, reduce_key='sample_size'):
    results = {}
    for elem in logging_outputs:
        for key in elem:
            if key not in results:
                results[key] = []
            if elem[key] is None:
                continue
            results[key].append(elem[key])

    logging_outputs = {}
    for key in results:
        if isinstance(results[key][0], torch.Tensor):
            logging_outputs[key] = torch.stack(results[key], dim=0).sum()
        else:
            logging_outputs[key] = torch.tensor(results[key]).sum()
    
    if reduce_key:
        reduce_size = logging_outputs[reduce_key]
        logging_outputs = {
            key: val / reduce_size for key, val in logging_outputs.items()
            if key != reduce_key
        }

    return logging_outputs


def skip_validation(args, extr_outputs):
    combined_nll_loss = torch.cat([_["combined_nll_loss"] for _ in extr_outputs], dim=0)
    nmt_nll_loss = torch.cat([_["nmt_nll_loss"] for _ in extr_outputs], dim=0)
    knn_alpha = torch.cat([_["knn_alpha"] for _ in extr_outputs], dim=0)
    knn_lambda = torch.cat([_["knn_lambda"] for _ in extr_outputs], dim=0)
    
    skip_num = int(knn_alpha.size(0) * args.validation_skip_percent)
    skip_index = knn_alpha.sort().indices[:skip_num]

    skip_nll_loss = combined_nll_loss.clone()
    skip_nll_loss[skip_index] = nmt_nll_loss[skip_index]

    outputs = {
        "nmt_ppl": torch.exp(nmt_nll_loss.mean()),
        "comb_ppl": torch.exp(combined_nll_loss.mean()),
        "skip_ppl": torch.exp(skip_nll_loss.mean()),
        "skip_threshold": knn_alpha[skip_index[-1]]
    }
    return outputs


def train(args, model, optimizer, criterion, train_dataloader, valid_dataloader):
    best_loss = 1e5
    best_ppl = 1e5
    best_skip_ppl = 1e5
    begin_time = time.time()
    patience = args.patience

    for epoch in range(args.max_epoch):
        logger.info(f'begin training for epoch {epoch + 1}')

        train_logging_outputs = []
        for sample in tqdm(train_dataloader):
            model.train()
            optimizer.zero_grad()

            loss, logging_output, _ = criterion(model, sample)
            train_logging_outputs.append(logging_output)
            loss.backward()
            optimizer.step()
            del loss

        train_logging_outputs = gather_list_and_reduce(train_logging_outputs)
        logger.info(f'end of epoch {epoch + 1} (average epoch stats below)')
        train_logging_info = [f'{name}: {val:.3f}' for name, val in train_logging_outputs.items()]
        train_logging_info_str = ' | '.join(train_logging_info)
        logger.info(train_logging_info_str)

        logger.info(f'begin validation')
        valid_logging_outputs = []
        skip_extr_outputs = []
        with torch.no_grad():
            model.eval()
            for sample in valid_dataloader:
                _loss, logging_output, skip_extr = criterion(model, sample, skip_extr=True, validation=True)
                valid_logging_outputs.append(logging_output)
                skip_extr_outputs.append(skip_extr)
                del _loss

        valid_outputs = gather_list_and_reduce(valid_logging_outputs)

        if (epoch + 1) % 5 == 0:
            model.dump(f'{args.save_path}/checkpoint_{epoch + 1}.pt')

        patience_flag = True
        if best_loss > valid_outputs["loss"]:
            patience_flag = False
            best_loss = valid_outputs["loss"]
            model.dump(f'{args.save_path}/checkpoint_best.pt')
            patience = args.patience

        if patience_flag:
            patience -= 1

        logger.info(' | '.join(
            ['valid'] \
            + [f'{name}: {val:.3f}' for name, val in valid_outputs.items()] \
            + [f'best_loss: {best_loss:.3f}'] \
        ))

        if patience == 0:
            logger.info('early stop !')
            break
    
    logger.info(f'done training in {time.time() - begin_time:.1f} seconds')


def main():
    args = parse_args()
    args.output_projection_path = f'{args.base_model_path}/output_projection.pt'
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_list = ['query', 'target', 'knn_dist', 'knn_tgt', 'attn']
    dataset_size = 0
    valid_data_dict = {}
    for name in tqdm(data_list):
        valid_data_dict[name] = torch.load(f'{args.data_path}/{name}.pt')
        dataset_size = valid_data_dict[name].shape[0]

    shuffle_indexs = torch.randperm(dataset_size)
    for name in tqdm(data_list):
        valid_data_dict[name] = valid_data_dict[name][shuffle_indexs]


    split_point = int(0.9 * dataset_size)
    train_dataset = MeDataset(**valid_data_dict, right=split_point, move_data_to_gpu=args.move_data_to_gpu)
    valid_dataset = MeDataset(**valid_data_dict, left=split_point, move_data_to_gpu=args.move_data_to_gpu)
    all_datasets = MeDataset(**valid_data_dict, move_data_to_gpu=args.move_data_to_gpu)

    vocab = Dictionary.load(f'{args.base_model_path}/fairseq-vocab.txt')
    args.probability_dim = len(vocab)
    logger.info('finished loading data')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate,
    )

    # model & optimizer & criterion
    model = SkipCombiner(
        temperature=args.temperature,
        knn_lambda=args.knn_lambda,
        mode="train",
        max_k=args.knn_max_k,
        probability_dim=args.probability_dim,
        output_projection_path=args.output_projection_path,
        token_feature_path=args.data_path,
        with_adaptive=args.with_adaptive,
    ).cuda()
    logger.info(model)

    for name, param in model.named_parameters():
        if 'output_projection' in name:
            param.requires_grad = False
        
        if param.requires_grad:
            logger.info(f'{name}: {param.shape}')
    
    logger.info(
        "model params: {} (trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-8, betas=[0.9, 0.98])
    criterion = MeCriterion(
        label_smoothing=0.001,
        padding_idx=vocab.pad(),
        norm_coef=args.norm_coef,
        alpha_coef=args.alpha_coef,
        alpha_mode=args.alpha_mode,
        balance_coef=args.balance_coef,
    ).cuda()

    train(
        args=args,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader,
    )

if __name__ == "__main__":
    main()
