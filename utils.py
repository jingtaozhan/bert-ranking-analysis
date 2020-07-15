import os
import re
import random
import subprocess
import unicodedata
from collections import defaultdict


def is_num(s):
    return all('0'<=c<='9' for c in s)


def is_punctuation(s):
    def _is_char_punctuation(char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    return all(_is_char_punctuation(c) for c in s)


def get_period_idxes(tokens):
    period_idxes = [idx for idx, token in enumerate(tokens) 
        if token =="." and (
        idx==0 or not is_num(tokens[idx-1]) or 
        idx==len(tokens)-1 or not is_num(tokens[idx+1]))
    ]
    return period_idxes


def generate_rank(input_path, output_path):
    score_dict = defaultdict(list)
    for line in open(input_path):
        query_id, para_id, score = line.split("\t")
        score_dict[int(query_id)].append((float(score), int(para_id)))
    with open(output_path, "w") as outFile:
        for query_id, para_lst in score_dict.items():
            random.shuffle(para_lst)
            para_lst = sorted(para_lst, key=lambda x:x[0], reverse=True)
            for rank_idx, (score, para_id) in enumerate(para_lst):
                outFile.write("{}\t{}\t{}\n".format(query_id, para_id, rank_idx+1))


def eval_results(run_file_path,
        eval_script="./ms_marco_eval.py", 
        qrels="./data/msmarco-passage/qrels.dev.small.tsv" ):
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path])
    match = re.search('MRR @10: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr