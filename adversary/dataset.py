import torch
import numpy as np
from dataset import pack_tensor_2D, pack_tensor_3D, load_qrels
from utils import generate_rank, eval_results, get_period_idxes


def get_collate_function(tokenizer, mask_method):
    period_id = tokenizer.convert_tokens_to_ids([","])[0]
    def collate_function(batch):
        period_indexes_lst = [ get_period_idxes(
                tokenizer.convert_ids_to_tokens(
                x["passage_input_ids"])) for x in batch]

        ret_batch = {}
        if mask_method == "commas":
            for data, period_indexes in zip(batch, period_indexes_lst):
                for period_idx in period_indexes:
                    data['passage_input_ids'][period_idx] = period_id
        else:
            periods_flags_lst = [np.ones(
                    len(x['passage_input_ids']) + len(x['query_input_ids'])) for x in batch]
            for arr, periods_indexes, data in zip(periods_flags_lst, period_indexes_lst, batch):
                if len(periods_indexes) == 0:
                    continue
                arr[np.array(periods_indexes) + len(data['query_input_ids'])] = 0

            if mask_method == "token_mask":
                ret_batch['attention_mask'] = pack_tensor_2D(
                        periods_flags_lst, default=0, dtype=torch.int64)
            elif mask_method == "attention_mask":
                ret_batch['attention_mask_after_softmax'] = pack_tensor_2D(
                        periods_flags_lst, default=0, dtype=torch.int64)
            else:
                raise NotImplementedError()

        input_ids_lst = [x["query_input_ids"] + x["passage_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["passage_input_ids"]) 
            for x in batch]
        ret_batch["input_ids"] = pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64)
        ret_batch["token_type_ids"]= pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64)

        if "attention_mask" not in ret_batch:
            attention_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
            ret_batch['attention_mask'] = pack_tensor_2D(attention_mask_lst, default=0, dtype=torch.int64)
        
        qid_lst = [x['qid'] for x in batch]
        pid_lst = [x['pid'] for x in batch]
        return ret_batch, qid_lst, pid_lst
    return collate_function  