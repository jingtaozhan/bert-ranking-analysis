import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import re
import json
import argparse
import numpy as np
import subprocess
import traceback
from collections import defaultdict
from matplotlib import pyplot as plt
from utils import get_period_idxes


def compute_attr(filepath, filter_wrong):
    all_cls_attribues, all_sep_attributes = [], []
    all_query_attr, all_para_attr, all_periods_attributes = [], [], []
    
    for line in open(filepath):
        data = json.loads(line)
        prediction = data['prediction']
        if filter_wrong and prediction < 0.5:
            continue
        tokens, attritbutes = data['tokens'], data['attributes']
        cls_attributes= attritbutes[0]
        sep_idx = tokens.index("[SEP]")
        sep_attribute = attritbutes[sep_idx] + attritbutes[-1]
        query_attribute = sum(attritbutes[1:sep_idx])
        para_attribute = sum(attritbutes[sep_idx+1:-1])

        para_tokens = tokens[sep_idx+1:-1]
        period_idxes = get_period_idxes(para_tokens)
        period_attribute = sum(attritbutes[i+sep_idx+1] for i in period_idxes)
        all_periods_attributes.append(period_attribute)

        all_cls_attribues.append(cls_attributes)
        all_para_attr.append(para_attribute)
        all_query_attr.append(query_attribute)
        all_sep_attributes.append(sep_attribute)
    return (np.mean(all_cls_attribues), np.mean(all_sep_attributes), 
        np.mean(all_query_attr), np.mean(all_para_attr), np.mean(all_periods_attributes))


def eval_segment(data_dir):
    x_arr = list(range(12))
    cls_arr, sep_arr, query_arr, para_arr, period_arr = [], [], [], [], []
    for x in x_arr:
        cls_attr, sep_attr, query_attr, para_attr, period_attr = compute_attr(
            f"{data_dir}/layer_{x}.json", filter_wrong=True
        )
        cls_arr.append(cls_attr)
        sep_arr.append(sep_attr)
        query_arr.append(query_attr)
        para_arr.append(para_attr)
        period_arr.append(period_attr)

    return x_arr, cls_arr, sep_arr, query_arr, para_arr, period_arr    


def draw(model_results_lst, save_path, x_label, titles):
    # Pretty colors
    colors = [
        "#e74c3c" ,#RED =
        "#f39c12",#ORANGE = 
        "#9b59b6",#PURPLE =
        "#159d82",#SEA =
        "#3498db", #BLUE =
        "#95a5a6",#GREY =
        "#59d98e",#GREEN :
        '#A52A2A', #BROWN   
        '#000080', #navy          
    ]
    plt.figure(figsize=(10, 4))
    for idx, (model_results, title) in enumerate(zip(model_results_lst, titles)):
        ax = plt.subplot(1, 2, idx+1)
        for idx, (label, (x_arr, y_arr)) in enumerate(model_results.items()):
            ax.plot(x_arr, y_arr, label=label, color=colors[idx])
        ax.legend(loc="best")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Attribute")
        ax.set_title(title)
    plt.savefig(save_path, bbox_inches='tight')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./data/attribution.pdf")
    parser.add_argument("--data_root", type=str, default="./data/attribution")
    args = parser.parse_args()

    lst = []
    for segment in ['query', 'para']:
        x_arr, cls_arr, sep_arr, query_arr, para_arr, period_arr = eval_segment(
            f"{args.data_root}/{segment}"
        )
        eval_dict = {
            "[CLS]":(x_arr, cls_arr),
            "[SEP]":(x_arr, sep_arr),
            "Query":(x_arr, query_arr),
            "Document":(x_arr, para_arr),
            "Periods": (x_arr, period_arr),
        }
        lst.append(eval_dict)
    draw(lst, args.output_path, x_label="layer", titles=["Empty Query Baseline", "Empty Document Baseline"])
