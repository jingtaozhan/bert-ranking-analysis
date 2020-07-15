import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import namedtuple, defaultdict
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import CollectionDataset, load_queries, pack_tensor_2D, pack_tensor_3D


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


class TopNDataset(Dataset):
    def __init__(self, topN_path, tokenizer, mode, msmarco_dir, collection_memmap_dir, tokenize_dir,
            max_query_length, max_seq_length):
        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.mode = mode
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.qids, self.pids = [], []
        for line in open(topN_path):
            qid, pid, _ = line.split()
            self.qids.append(int(qid))
            self.pids.append(int(pid))

    def __len__(self):  
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, passage_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        passage_input_ids = passage_input_ids[:self.max_seq_length - 1 - len(query_input_ids)]
        passage_input_ids = passage_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "passage_input_ids": passage_input_ids,
            "qid": qid,
            "pid" : pid,
        }
        return ret_val


def mask_method(mask_target, input_ids, mid_sep_idx):
    att_mask = np.ones((len(input_ids), len(input_ids)), dtype=np.int32)    
    para_begin, para_end = mid_sep_idx + 1, len(input_ids)
    query_begin, query_end = 1, mid_sep_idx
    mask_all = mask_target == "mask_both_query_para"
    if mask_target == "mask_para_from_query" or mask_all:
        att_mask[para_begin:para_end, :para_begin] = 0
        att_mask[para_begin:para_end, para_end:] = 0
    if mask_target == "mask_query_from_para" or mask_all:
        att_mask[query_begin:query_end, :query_begin] = 0
        att_mask[query_begin:query_end, query_end:] = 0
    return att_mask


def get_collate_function(mask_target):
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["passage_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["passage_input_ids"]) 
            for x in batch]
        attention_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
        att_mask_after_softmax_lst = [
            mask_method(mask_target, input_ids, len(x["query_input_ids"])-1)
            for input_ids, x in zip(input_ids_lst, batch)]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "attention_mask": pack_tensor_2D(attention_mask_lst, default=0, dtype=torch.int64),
            "attention_mask_after_softmax": pack_tensor_3D(att_mask_after_softmax_lst, default=1, dtype=torch.int64)
        }
        qid_lst = [x['qid'] for x in batch]
        pid_lst = [x['pid'] for x in batch]
        return data, qid_lst, pid_lst
    return collate_function  


if __name__ == "__main__":
    pass
    
