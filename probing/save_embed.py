import os
import glob
import torch
import random
import logging
import argparse
import zipfile
import json
import unicodedata
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer

from modeling import MonoBERT
from utils import get_period_idxes, is_num, is_punctuation
from mask.dataset import TopNDataset
from dataset import get_collate_function

def get_indexes(stop_words_set, tokenizer, input_tokens, mid_sep_index, load_key):
    if load_key == "cls":
        return [0]
    elif load_key == "all_query_tokens": # query
        return list(range(1, mid_sep_index))
    elif load_key == "seps":
        return [mid_sep_index, len(input_tokens)-1]
    elif load_key == "rand_passage_tokens": # document
        all_idxes = list(range(mid_sep_index+1, len(input_tokens)-1))
        np.random.shuffle(all_idxes)
        return all_idxes[:10]
    elif load_key == "periods_in_passage": # period
        para_tokens = input_tokens[mid_sep_index+1:]
        period_idxes = get_period_idxes(para_tokens)
        period_idxes = [idx+mid_sep_index+1 for idx in period_idxes]
        assert all(input_tokens[idx]=="." for idx in period_idxes)
        return period_idxes
    elif load_key == "stopwords_in_passage":
        para_tokens = input_tokens[mid_sep_index+1:]
        stopword_idxes = [idx+mid_sep_index+1 for idx, token in enumerate(para_tokens) 
            if token in stop_words_set
        ]
        np.random.shuffle(stopword_idxes)
        stopword_idxes = stopword_idxes[:15]
        return stopword_idxes
    elif load_key == "query_tokens_in_passage":
        query_tokens_set = set(input_tokens[1:mid_sep_index]) - stop_words_set
        para_tokens = input_tokens[mid_sep_index+1:]
        query_item_idxes = [idx+mid_sep_index+1 for idx, token in enumerate(para_tokens) 
            if token in query_tokens_set
        ]
        return query_item_idxes[:20]
    else:
        raise NotImplementedError()


def load_stopwords(idf_path):
    idf_raw_dict = json.load(open(idf_path))
    for k in idf_raw_dict.keys():
        if is_punctuation(k) or is_num(k):
            idf_raw_dict[k] = 0
    wordlst = list(idf_raw_dict.items())
    wordlst = sorted(wordlst, key=lambda x:x[1])[::-1]
    stop_words_set = set(x[0] for x in wordlst[1:51]) # ALL_DOC_NUM is not a token
    return stop_words_set


def evaluate(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for key in args.keys:
        key_dir = f"{args.output_dir}/{key}"
        for layer_idx in range(model.config.num_hidden_layers+1):
            layer_dir = f"{key_dir}/{layer_idx}"
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)

    stop_words_set = load_stopwords(args.idf_path)

    eval_dataset = TopNDataset(args.rank_file, tokenizer, 
        args.mode, args.msmarco_dir, 
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
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    for batch, qids, pids in tqdm(eval_dataloader):
        model.eval()
        batch = {k:v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            all_layers_hidden_states = model(**batch)[1]
            all_layers_hidden_states = [h.detach().cpu().numpy()
                for h in all_layers_hidden_states]
            save_to_disk(tokenizer, stop_words_set, all_layers_hidden_states, 
                args, qids, pids, batch)
                

def save_to_disk(tokenizer, stop_words_set, all_layers_hidden_states, args, qids, pids, batch):
    for idx, (qid, pid, input_ids, token_type_ids, attention_mask) in enumerate(zip(
        qids, pids, batch['input_ids'].cpu().numpy(),
        batch['token_type_ids'], batch['attention_mask']
    )):
        input_length = torch.sum(attention_mask).item()
        input_ids = input_ids[:input_length].tolist()
        token_type_ids = token_type_ids[:input_length]
        mid_sep_idx = torch.sum(token_type_ids==0).item()-1
        assert input_ids[mid_sep_idx] == tokenizer.sep_token_id
        assert input_ids[-1] == tokenizer.sep_token_id
        assert input_ids[0] == tokenizer.cls_token_id
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for key in args.keys:
            idxes = get_indexes(stop_words_set, tokenizer, input_tokens, mid_sep_idx, key)
            for layer_idx, all_hidden_states in enumerate(all_layers_hidden_states):
                save_hidden_states = all_hidden_states[idx][idxes].astype(np.float16)
                save_path = f"{args.output_dir}/{key}/{layer_idx}/{qid}-{pid}.npy"
                np.save(save_path, save_hidden_states)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--keys", type=str, nargs="+", default=[
        "cls", "seps", "periods_in_passage", "all_query_tokens", 
        "rand_passage_tokens", "stopwords_in_passage", "query_tokens_in_passage"])
    parser.add_argument("--rank_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "dev.small"], required=True)
    parser.add_argument("--idf_path", type=str, default="./data/wordpiece.idf.json")

    parser.add_argument("--output_root", type=str, default="./data/probing/embed")
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

    args.output_dir = f"{args.output_root}/{args.mode}"
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    print("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = BertConfig.from_pretrained(f"{args.model_path}/bert_config.json")
    config.output_hidden_states = True
    model = MonoBERT.from_pretrained(f"{args.model_path}/model.ckpt-100000", 
        from_tf=True, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    model.to(args.device)

    print("Training/evaluation parameters %s", args)

    # Evaluation
    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()

