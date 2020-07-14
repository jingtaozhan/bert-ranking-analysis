import os
import glob
import torch
import random
import logging
import argparse
import zipfile
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertTokenizer)

from modeling import MonoBERT
from dataset import RelevantDataset, get_collate_function

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_dataset = RelevantDataset(tokenizer, "dev.small", args.msmarco_dir, 
        args.collection_memmap_dir, args.tokenize_dir, 
        args.max_query_length, args.max_seq_length)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
        collate_fn=get_collate_function())

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_scores, all_ids = [], []
    for batch, qids, pids in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = {k:v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            attentions = model(**batch)[1]

            for layer_id, layer_attentions in enumerate(attentions):
                attention_dir = os.path.join(eval_output_dir, "layer_{}".format(layer_id+1))
                if not os.path.exists(attention_dir):
                    os.makedirs(attention_dir)
                for idx, attention in enumerate(layer_attentions):
                    length = torch.sum(batch['attention_mask'][idx]).detach().cpu().item()
                    query_id, para_id = qids[idx], pids[idx]
                    attention = attention[:, :length, :length].detach().cpu().numpy()
                    file_path = os.path.join(attention_dir, "{}-{}.npy".format(query_id, para_id))
                    np.save(file_path, np.array(attention, dtype=np.float16))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--output_dir", type=str, default="./data/attention")
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--model_path", type=str, default="./data/BERT_Base_trained_on_MSMARCO")

    ## Other parameters
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()
    
    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = BertConfig.from_pretrained(f"{args.model_path}/bert_config.json")
    config.output_attentions = True
    model = MonoBERT.from_pretrained(f"{args.model_path}/model.ckpt-100000", 
        from_tf=True, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Evaluation
    evaluate(args, model, tokenizer)


