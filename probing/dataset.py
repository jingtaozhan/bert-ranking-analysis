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

from dataset import pack_tensor_3D, load_qrels


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

  
class ProbDataset(Dataset):
    def __init__(self, embd_data_dir, msmarco_dir, mode, max_token_num):
        qrels = defaultdict(list)
        for qid, pid in zip(*load_qrels(f"{msmarco_dir}/qrels.{mode}.tsv")):
            qrels[qid].append(pid)
        self.qrels = dict(qrels)
        self.data_lst = os.listdir(embd_data_dir)
        self.embd_data_dir = embd_data_dir
        self.max_token_num = max_token_num

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, item):
        filename = self.data_lst[item]
        qid, pid = filename[:-len(".npy")].split("-")
        qid, pid = int(qid), int(pid)
        hidden_states = np.load(f"{self.embd_data_dir}/{filename}")     
        hidden_states = hidden_states[:self.max_token_num, :]
        ret_val = {
            "hidden_states": hidden_states, 
            "label": 1 if pid in self.qrels[qid] else 0,
            "qid": qid,
            "pid": pid
        }
        return ret_val


def get_collate_function():
    def collate_function(batch):
        hidden_states_lst = [x["hidden_states"] for x in batch]
        label_lst = [[x["label"]] for x in batch]
        data = {
            "hidden_states": pack_tensor_3D(hidden_states_lst, default=0, dtype=torch.float32),
            "labels": torch.LongTensor(label_lst),
        }
        qid_lst = [x['qid'] for x in batch]
        pid_lst = [x['pid'] for x in batch]
        return data, qid_lst, pid_lst
    return collate_function  
