import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager

def run_certain_layer(gpu_queue, args, trunc_layer, outputpath):
    gpu = gpu_queue.get_nowait()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from transformers import BertTokenizer, BertConfig
    from attribution.modeling import TruncMonoBERT
    from attribution.ig_utils import get_ig_attributions, visualize_token_attrs
    from dataset import RelevantDataset
    
    config = BertConfig.from_pretrained(f"{args.model_path}/bert_config.json")
    config.output_hidden_states = True
    model = TruncMonoBERT.from_pretrained(f"{args.model_path}/model.ckpt-100000", 
        from_tf=True, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    dataset = RelevantDataset(tokenizer, "dev.small", args.msmarco_dir, 
        args.collection_memmap_dir, args.tokenize_dir, 
        args.max_query_length, args.max_seq_length)
        
    device = torch.device("cuda")
    assert torch.cuda.device_count() == 1
    model.to(device)
    with open(outputpath, 'w') as ig_output:
        for idx, data in enumerate(tqdm(dataset, desc=f"layer:{trunc_layer}")):   
            ig, error_percentage, baseline_prediction, prediction, num_reps = \
                get_ig_attributions(
                    trunc_layer=trunc_layer,
                    model=model, tokenizer=tokenizer, 
                    query_tokens=tokenizer.convert_ids_to_tokens(data['query_input_ids'][1:-1]), 
                    para_tokens=tokenizer.convert_ids_to_tokens(data['passage_input_ids'][:-1]), 
                    label=1, batch_size=args.batch_size,
                    device=device, attr_segment=args.attr_segment, 
                    max_reps=args.max_reps, begin_num_reps=args.begin_num_reps,
                    max_allowed_error=args.max_allowed_error,
                    max_query_length=args.max_query_length, 
                    max_seq_length=args.max_seq_length,
                    debug=args.debug)
            tokens, attributes = ig['outputs']
            if args.show_details:
                print("idx:{}/{} error:{:.1f} reps:{}".format(
                    idx, len(dataset), error_percentage, num_reps))
                color_text = visualize_token_attrs(tokens, attributes)
                print(color_text)
            data = {
                "tokens":tokens,
                "attributes":attributes,
                "qid":data['qid'], 
                "pid":data['pid'],
                "error_percentage":error_percentage, 
                "baseline_prediction":baseline_prediction, 
                "prediction":prediction, 
                "num_reps":num_reps
            }
            write_line = json.dumps(data)
            ig_output.write("{}\n".format(write_line))
            #if idx > 100:
            #    break
    model, dataset = None, None
    gpu_queue.put_nowait(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--attr_segment", choices=["query", "para"], required=True)
    parser.add_argument("--trunc_layers", nargs="+", type=int, default=list(range(13)))
    parser.add_argument("--output_root", type=str, default="./data/attribution")

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--begin_num_reps", type=int, default=4)
    parser.add_argument("--max_reps", type=int, default=30)
    parser.add_argument("--max_allowed_error", type=float, default=5.0)
    
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--model_path", type=str, default="./data/BERT_Base_trained_on_MSMARCO")
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_details", action="store_true")
    
    args = parser.parse_args()

    args.output_dir = f"{args.output_root}/{args.attr_segment}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    manager = Manager()
    gpu_queue = manager.Queue()
    for gpu in args.gpus:
        gpu_queue.put_nowait(gpu)

    pool = Pool(len(args.gpus))
    arguments = [(gpu_queue, args, trunc_layer, 
            f"{args.output_dir}/layer_{trunc_layer}.json") 
            for trunc_layer in args.trunc_layers]
    pool.starmap(run_certain_layer, arguments)
    pool.close()
    pool.join()