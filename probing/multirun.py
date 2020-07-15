import os
import json
import random
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, Manager    
    

def call_runprob(gpu_queue, layer, key, run_cmd):
    gpu = gpu_queue.get_nowait()
    subprocess.check_call(["python", "-m", "probing.runprob", "--layer", str(layer),
        "--key", key, run_cmd, "--gpu", str(gpu)])
    gpu_queue.put_nowait(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--keys", type=str, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--train_rank_file", type=str, default="./data/sample.train.tsv")
    parser.add_argument("--eval_rank_file", type=str, default="./data/anserini.dev.small.top100.tsv")
    parser.add_argument("--save_train_embed", action="store_true")
    parser.add_argument("--do_train", action="store_true")   
    parser.add_argument("--save_eval_embed", action="store_true")
    parser.add_argument("--do_eval", action="store_true") 
    args = parser.parse_args()

    if args.save_train_embed:
        subprocess.check_call(["python", "-m", "probing.save_embed", 
            "--keys"] + args.keys [+ "--rank_file", args.train_rank_file,
            "--mode", "train",
            "--gpu", str(args.gpus[0])])
    
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in args.gpus:
        gpu_queue.put_nowait(gpu)

    if args.do_train:
        pool = Pool(len(args.gpus))
        arguments = [(gpu_queue, layer, key, "--do_train") 
            for layer in args.layers for key in args.keys]
        random.shuffle(arguments)
        pool.starmap(call_runprob, arguments)
        pool.close()
        pool.join()
    
    if args.save_eval_embed:
        subprocess.check_call(["python", "-m", "probing.save_embed", 
            "--keys"] + args.keys + [ "--rank_file", args.eval_rank_file,
            "--mode", "dev.small",
            "--gpu", str(args.gpus[0])])
    
    if args.do_eval:
        pool = Pool(len(args.gpus))
        arguments = [(gpu_queue, layer, key, "--do_eval") 
            for layer in args.layers for key in args.keys]
        random.shuffle(arguments)
        pool.starmap(call_runprob, arguments)
        pool.close()
        pool.join()