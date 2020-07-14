import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class CollectionDataset:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        assert self.pids[item] == item
        return self.token_ids[item, :self.lengths[item]].tolist()


def load_queries(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/queries.{mode}.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_qrels(filepath):
    qids, pids = [], []
    for line in open(filepath):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        qids.append(qid)
        pids.append(pid)
    return qids, pids


class RelevantDataset(Dataset):
    def __init__(self, tokenizer, mode, msmarco_dir, collection_memmap_dir, tokenize_dir,
            max_query_length, max_seq_length):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.mode = mode
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.qids, self.pids = load_qrels(f"{msmarco_dir}/qrels.{mode}.tsv")

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
            "pid" : pid
        }
        return ret_val


def pack_tensor_2D(lstlst, default, dtype):
    batch_size = len(lstlst)
    length =  max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function():
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["passage_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["passage_input_ids"]) 
            for x in batch]
        attention_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "attention_mask": pack_tensor_2D(attention_mask_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        pid_lst = [x['pid'] for x in batch]
        return data, qid_lst, pid_lst
    return collate_function  


    
    