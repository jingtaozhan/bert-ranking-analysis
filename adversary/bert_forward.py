import os
import re
import json
import torch
import random
import logging
import argparse
import subprocess
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import BertConfig, BertTokenizer

from mask.dataset import TopNDataset
from dataset import RelevantDataset
from mask.modeling import MaskMonoBERT
from modeling import MonoBERT
from dataset import pack_tensor_2D, pack_tensor_3D, load_qrels
from utils import generate_rank, eval_results, get_period_idxes
import adversary.dataset as adverse_dataset
import dataset as origin_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.WARN)


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    if args.mask_method == "None":
        eval_dataset = TopNDataset(args.topN_file, tokenizer, "dev.small", args.msmarco_dir, 
            args.collection_memmap_dir, args.tokenize_dir, 
            args.max_query_length, args.max_seq_length)
        collate_func = origin_dataset.get_collate_function()
    else:
        eval_dataset = RelevantDataset(tokenizer, "dev.small", args.msmarco_dir, 
            args.collection_memmap_dir, args.tokenize_dir, 
            args.max_query_length, args.max_seq_length)
        collate_func = adverse_dataset.get_collate_function(tokenizer, args.mask_method)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, 
            collate_fn=collate_func)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    cnt = 0
    with open(args.output_score_path, 'w') as outputfile:
        for batch, qids, pids in tqdm(eval_dataloader):
            model.eval()
            batch = {k:v.to(args.device) for k, v in batch.items()}
            if args.mask_method == "attention_mask":
                batch['attention_mask_after_softmax_layer_set'] = list(range(model.config.num_hidden_layers))
            with torch.no_grad():
                outputs = model(**batch)
                scores = outputs[0].detach().cpu().numpy()
                for qid, pid, score in zip(qids, pids, scores[:, 1]):
                    outputfile.write(f"{qid}\t{pid}\t{score}\n")
            cnt += 1
            # if cnt > 1000:
            #     break


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--mask_method", type=str, choices=[
        "commas", "token_mask", "attention_mask", "None"], required=True)
    parser.add_argument("--output_dir", type=str, default="./data/adversary")
    parser.add_argument("--topN_file", type=str, default="./data/anserini.dev.small.top100.tsv")
    parser.add_argument("--qrel_file", type=str, default="./data/msmarco-passage/qrel.dev.small.tsv")
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--model_path", type=str, default="./data/BERT_Base_trained_on_MSMARCO")

    ## Other parameters
    parser.add_argument("--gpu", default=None, type=str, required=False)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_score_path = f"{args.output_dir}/{args.mask_method}.score.tsv"

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.info(args)
    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.info("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = BertConfig.from_pretrained(f"{args.model_path}/bert_config.json")
    model_class = MaskMonoBERT if args.mask_method == "attention_mask" else MonoBERT
    model = model_class.from_pretrained(f"{args.model_path}/model.ckpt-100000", 
        from_tf=True, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation
    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()
