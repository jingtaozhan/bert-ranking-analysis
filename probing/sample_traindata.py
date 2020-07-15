import random
import argparse
from tqdm import tqdm
from collections import defaultdict

from dataset import load_qrels

def sample(topN_path, qrel_path, output_path):
    qrels = defaultdict(list)
    for qid, pid in zip(*load_qrels(qrel_path)):
        qrels[qid].append(pid)
    topN = defaultdict(set)
    for line in tqdm(open(topN_path)):
        qid, pid, _ = line.split()
        topN[int(qid)].add(int(pid))
    with open(output_path, 'w') as outputfile:
        for qid, pidset in tqdm(topN.items()):
            pos_pid = random.choice(qrels[qid])
            neg_pid = random.choice(list(pidset-set(qrels[qid])))
            outputfile.write(f"{qid}\t{neg_pid}\t0\n")
            outputfile.write(f"{qid}\t{pos_pid}\t1\n")       


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topN_path", type=str, default="./data/anserini.train.top100.tsv")
    parser.add_argument("--qrel_path", type=str, default="./data/msmarco-passage/qrels.train.tsv")
    parser.add_argument("--output_path", type=str, default="./data/sample.train.tsv")
    args = parser.parse_args()

    sample(args.topN_path, args.qrel_path, args.output_path)