import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict
from transformers import BertTokenizer

from dataset import RelevantDataset
from utils import get_period_idxes


def one_layer_attention_pattern(tokenizer, attention_dir, dataset, id2pos):
    avg_attns = {
        k: np.zeros(12, dtype=np.float64) for k in [
            "query->query", "query->para", "para->query", "para->para", 
            "interactive", "representative", "[CLS]", "[SEP]", "periods"
        ]
    }
    n_docs = 0
    for filename in tqdm(os.listdir(attention_dir)):
        qid, pid = filename.rstrip(".npy").split("-")
        qid, pid = int(qid), int(pid)
        data = dataset[id2pos[(qid, pid)]]
        assert qid == data['qid'] and pid == data['pid']
        query_input_ids = data["query_input_ids"]
        passage_input_ids = data["passage_input_ids"]
        attention_map = np.load(f"{attention_dir}/{filename}")
        n_tokens = len(query_input_ids) + len(passage_input_ids)
        assert len(attention_map[0]) == n_tokens
        mid_sep_idx = len(query_input_ids) - 1

        seps, clss, periods = (np.zeros(n_tokens) for _ in range(3))
        clss[0], seps[mid_sep_idx], seps[-1] = 1, 1, 1 
        seq_tokens = tokenizer.convert_ids_to_tokens(query_input_ids + passage_input_ids)
        for position, token in enumerate(seq_tokens):
            if (token == "." and (not is_num(seq_tokens[position-1])
                or not is_num(seq_tokens[position+1]))):
                periods[position] = 1
        assert seq_tokens[0] == "[CLS]" and seq_tokens[mid_sep_idx] == "[SEP]"
        selectors = {
            "query->query": attention_map[:, 1:mid_sep_idx, 1:mid_sep_idx],
            "para->query": attention_map[:, 1:mid_sep_idx, mid_sep_idx+1:n_tokens-1],
            "query->para": attention_map[:, mid_sep_idx+1:n_tokens-1, 1:mid_sep_idx],
            "para->para": attention_map[:, mid_sep_idx+1:n_tokens-1, mid_sep_idx+1:n_tokens-1],
        }

        for key, selector in selectors.items():
            avg_attns[key] += selector.sum(-1).mean(-1)
        avg_attns["[SEP]"] += (attention_map * seps).sum(-1).mean(-1)
        avg_attns["[CLS]"] += (attention_map * clss).sum(-1).mean(-1)
        avg_attns["periods"] += (attention_map * periods).sum(-1).mean(-1)
        n_docs += 1

        # if n_docs > 20:
        #     break
    for key in avg_attns.keys():
        avg_attns[key] /= n_docs
    return avg_attns


def get_data_points(head_data):
    xs, ys, avgs = [], [], []
    for layer in range(12):
        for head in range(12):
            ys.append(head_data[layer][head])
            xs.append(1 + layer)
        avgs.append(head_data[layer].mean())
    return xs, ys, avgs


def add_line(avg_attns, key, ax, color, label, plot_avgs=True):
    xs, ys, avgs = get_data_points(avg_attns[key])
    ax.scatter(xs, ys, s=12, label=label, color=color)
    if plot_avgs:
        ax.plot(1 + np.arange(len(avgs)), avgs, color=color)
    ax.legend(loc="best")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg. Attention")


def plot_curve(avg_attns):
    # Pretty colors
    BLACK = "k"
    GREEN = "#59d98e"
    SEA = "#159d82"
    BLUE = "#3498db"
    PURPLE = "#9b59b6"
    GREY = "#95a5a6"
    RED = "#e74c3c"
    ORANGE = "#f39c12"

    plt.figure(figsize=(11, 4))
    ax = plt.subplot(1, 2, 1)
    for key, color, label in [
        ("[CLS]", RED, "[CLS]"),
        ("[SEP]", BLUE, "[SEP]"),
        ("periods", PURPLE, "Periods"), ]:
        add_line(avg_attns, key, ax, color, label)
    
    ax = plt.subplot(1, 2, 2)
    for key, color, label in [
        ("query->query", RED, r"query$\rightarrow$query"),
        ("query->para", BLUE, r"query$\rightarrow$doc"),
        ("para->query", PURPLE, r"doc$\rightarrow$query"), 
        ("para->para", GREEN, r"doc$\rightarrow$doc")]:
        add_line(avg_attns, key, ax, color, label, plot_avgs=True)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_path", type=str, default="./data/avg_att.pdf")
    parser.add_argument("--attention_dir", type=str, default="./data/attention")
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--output_dir", type=str, default="./data/attention")
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--model_path", type=str, default="./data/BERT_Base_trained_on_MSMARCO")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    dataset = RelevantDataset(tokenizer, "dev.small", args.msmarco_dir, 
        args.collection_memmap_dir, args.tokenize_dir, 
        args.max_query_length, args.max_seq_length)
    id2pos = {(qid, pid):i for i, (qid, pid) in enumerate(zip(dataset.qids, dataset.pids))}

    all_results = defaultdict(list)
    for layer_idx in tqdm(range(12)):
        avg_attns = one_layer_attention_pattern(tokenizer,
            attention_dir=f"{args.attention_dir}/layer_{layer_idx+1}",
            dataset=dataset, id2pos=id2pos)
        for key, value in avg_attns.items():
            all_results[key].append(value)

    plot_curve(all_results)
    plt.savefig(args.output_path, bbox_inches='tight')