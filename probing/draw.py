import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
import re
import copy
import numpy as np
import subprocess
import argparse
from matplotlib import pyplot as plt


def draw(model_results, save_path):
    # Pretty colors
    colors = [
        "#59d98e",#GREEN :
        "#f39c12",#ORANGE = 
        "#159d82",#SEA =
        "#3498db", #BLUE =
        "#9b59b6",#PURPLE =
        "#95a5a6",#GREY =
        "#e74c3c" ,#RED =
        '#A52A2A', #BROWN   
        '#000080', #navy          
    ]
    plt.figure(figsize=(8, 5))
    color_idx = 0
    for label, (xs, ys) in model_results.items():
        plt.plot(xs, ys, label=label, color=colors[color_idx])
        color_idx += 1
    plt.legend(loc="best")
    plt.xlabel("Layer")
    plt.ylabel("MRR@10")
    plt.savefig(save_path, bbox_inches='tight')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/probing/eval")
    parser.add_argument("--output_path", type=str, default="./data/probing.pdf")
    args = parser.parse_args()

    model_results = {}
    for folder_name, label in [
            ("all_query_tokens", "Query"),
            ("cls", "[CLS]"),
            ("seps", "[SEP]"),
            ("periods_in_passage", "Periods (Doc)"),
            ("rand_passage_tokens", "Random (Doc)"),
            ("stopwords_in_passage", "Stop Words (Doc)"),
            ("query_tokens_in_passage", "Query Terms (Doc)")]:
        points = []
        for filename in os.listdir(f"{args.data_root}/{folder_name}"):
            matchobj = re.match(r'layer_([\d.]+).rank.tsv.([\d.]+)', filename)
            if matchobj:
                points.append((int(matchobj.group(1)), float(matchobj.group(2))))
        points = sorted(points)
        x_arr, y_arr = [x[0] for x in points], [x[1] for x in points]
        model_results[label] = (x_arr, y_arr)

    draw(model_results, args.output_path)


