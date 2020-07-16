import os
import re
import random
import argparse
import subprocess
import unicodedata
from tqdm import tqdm
from collections import defaultdict
from utils import eval_results, generate_rank

def read_scores(filepath):
    score_dict = defaultdict(list)
    for line in open(filepath):
        query_id, para_id, score = line.split("\t")
        score_dict[(int(query_id), int(para_id))] = float(score)
    return dict(score_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_methods", type=str, nargs="+",
        default=["commas", "token_mask", "attention_mask", "None"])
    parser.add_argument("--input_dir", type=str, default="./data/adversary")
    parser.add_argument("--output_dir", type=str, default="./data/adversary")
    args = parser.parse_args()
    
    origin_scores = read_scores(f"{args.input_dir}/None.score.tsv")
    for mask_method in args.mask_methods:
        new_scores = read_scores(f"{args.input_dir}/{mask_method}.score.tsv")
        for key, score in new_scores.items():
            if key in origin_scores:
                origin_scores[key] = score
        temp_score_path = f"{args.output_dir}/temp.{mask_method}.score.tsv"
        assert not os.path.exists(temp_score_path)
        with open(temp_score_path, "w") as outFile:
            for (qid, pid), score in origin_scores.items():
                outFile.write(f"{qid}\t{pid}\t{score}\n")
        output_rank_path = f"{args.output_dir}/{mask_method}.rank.tsv"
        generate_rank(temp_score_path, output_rank_path)
        subprocess.check_call(["rm", temp_score_path])
        mrr = eval_results(output_rank_path)
        abs_output_rank_path = os.path.abspath(output_rank_path)
        rank_with_mrr_path = f"{abs_output_rank_path}.{mrr:.3f}"
        if not os.path.exists(rank_with_mrr_path):
            subprocess.check_call(["ln", "-s", abs_output_rank_path, rank_with_mrr_path])
        print(mask_method, "MRR@10:", mrr)
        
    