import os
import json
import random
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, Manager    
    

def call(gpu_queue, mask_target, mask_layer_num):
    gpu = gpu_queue.get_nowait()
    subprocess.check_call(["python", "-m", "mask.bert_forward", 
        "--mask_target", mask_target, 
        "--mask_layer_num", str(mask_layer_num), 
        "--gpu", str(gpu)])
    gpu_queue.put_nowait(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--mask_targets", type=str, nargs="+", 
        default=["mask_both_query_para", "mask_para_from_query", "mask_query_from_para"])
    parser.add_argument("--mask_layer_nums", type=int, nargs="+",
        default=list(range(12)))
    
    args = parser.parse_args()
    
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in args.gpus:
        gpu_queue.put_nowait(gpu)

    pool = Pool(len(args.gpus))
    arguments = [(gpu_queue, mask_target, mask_layer_num)
            for mask_target in args.mask_targets
            for mask_layer_num in args.mask_layer_nums]
    random.shuffle(arguments)
    pool.starmap(call, arguments)
    pool.close()
    pool.join()