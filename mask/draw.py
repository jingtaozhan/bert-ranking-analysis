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

def draw(model_results, save_path):
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
    plt.figure(figsize=(7, 4))
    for idx, (label, (x_arr, y_arr)) in enumerate(model_results.items()):
        plt.plot(x_arr, y_arr, label=label, color=colors[idx])
    plt.legend(loc="best")
    plt.xlabel("layer")
    plt.ylabel("MRR@10")
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/mask")
    parser.add_argument("--output_path", type=str, default="./data/mask.pdf")
    args = parser.parse_args()
    mask_target_lst = ["mask_both_query_para", "mask_para_from_query", "mask_query_from_para"]
    label_lst = ["mask both directions", r"mask query$\rightarrow$doc", r"mask doc$\rightarrow$query"]
    model_results = dict()
    for label, mask_target in zip(label_lst, mask_target_lst):
        data_dir = f"{args.data_root}/{mask_target}"
        points = []
        for filename in os.listdir(data_dir):
            matchobj = re.match(r'layer_([\d.]+).rank.tsv.([\d.]+)', filename)
            if matchobj:
                points.append((int(matchobj.group(1)), float(matchobj.group(2))))
        points = sorted(points)
        x_arr, y_arr = [x[0] for x in points], [x[1] for x in points]
        model_results[label] = (x_arr, y_arr)
    draw(model_results, args.output_path)
        
