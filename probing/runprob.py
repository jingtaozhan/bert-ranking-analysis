# coding=utf-8
import argparse
import logging
import os
import random
import numpy as np
import torch
import subprocess
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange

from probing.dataset import ProbDataset, get_collate_function
from probing.modeling import EmbeddingProb
from utils import generate_rank, eval_results


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s-%(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model):
    """ Train the model """
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
   
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    train_dataset = ProbDataset(f"{args.embd_root}/train/{args.key}/{args.layer}", 
            args.msmarco_dir, "train", args.max_token_num)

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        pin_memory=False, batch_size=args.train_batch_size, 
        collate_fn=get_collate_function(),
        num_workers=args.data_num_workers)
    
    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
        num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, _, _) in enumerate(epoch_iterator):
            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.train()
            loss = model(**batch)[0]
            loss.backward()
            tr_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                avg_loss = (tr_loss - logging_loss)/args.logging_steps
                tb_writer.add_scalar('loss', avg_loss, global_step)
                logging_loss = tr_loss
            '''
            if global_step > 3000:
                print("debug")
                break
            '''
    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model):
    eval_dataset = ProbDataset(f"{args.embd_root}/dev.small/{args.key}/{args.layer}", 
            args.msmarco_dir, "dev.small", args.max_token_num)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
        pin_memory=False, batch_size=args.eval_batch_size, 
        collate_fn=get_collate_function(),
        num_workers=args.data_num_workers)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    output_score_path = f"{args.eval_output_dir}/layer_{args.layer}.score.tsv"
    output_rank_path = f"{args.eval_output_dir}/layer_{args.layer}.rank.tsv"
    with open(output_score_path, 'w') as outputfile:
        for batch_idx, (batch, qids, pids) in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            del batch['labels']
            batch = {k:v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                softmax_logits = model(**batch)[0].detach().cpu().numpy()
                scores = softmax_logits[:, 1]
                for idx, (qid, pid, score) in enumerate(zip(qids, pids, scores)):
                    outputfile.write(f"{qid}\t{pid}\t{score}\n")
    generate_rank(output_score_path, output_rank_path)
    mrr = eval_results(output_rank_path)
    abs_output_rank_path = os.path.abspath(output_rank_path)
    mrr_ln_path = f"{abs_output_rank_path}.{mrr:.3f}"
    subprocess.check_call(["ln", "-s", abs_output_rank_path, mrr_ln_path])


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--key", type=str, choices=[
        "cls", "seps", "periods_in_passage", "all_query_tokens", 
        "rand_passage_tokens", "stopwords_in_passage", "query_tokens_in_passage"],
        required=True)

    parser.add_argument("--max_token_num", type=int, default=20)
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--eval_output_root", type=str, default="./data/probing/eval")
    parser.add_argument("--model_output_root", type=str, default="./data/probing/models")
    parser.add_argument("--log_root", type=str, default="./data/probing/log")
    parser.add_argument("--embd_root", type=str, default="./data/probing/embed")

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')

    parser.add_argument("--gpu", default=None, type=str, required=False)

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--data_num_workers", default=10, type=int)

    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument('--logging_steps', type=int, default=1000)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    assert args.n_gpu == 1

    args.device = device

    args.eval_output_dir = f"{args.eval_output_root}/{args.key}"
    args.model_output_dir = f"{args.model_output_root}/{args.key}"
    args.log_dir = f"{args.log_root}/{args.key}/{args.layer}"
    
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    if not os.path.exists(args.eval_output_dir):
        os.makedirs(args.eval_output_dir)

    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    # Set seed
    set_seed(args)

    model = EmbeddingProb(hidden_size=args.hidden_size, 
        max_token_num=args.max_token_num, 
        dropout_prob=args.dropout_prob)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    model_save_path = f"{args.model_output_dir}/layer_{args.layer}_model.bin"
    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model)
        logger.info(f"Saving model checkpoint to {model_save_path}")
        torch.save(model.state_dict(), model_save_path)

    # Load a trained model 
    if args.do_eval:
        print(model_save_path)
        model = EmbeddingProb(args.hidden_size, args.max_token_num, dropout_prob=0.0)
        model.load_state_dict(torch.load(model_save_path))
        model.to(args.device)
        evaluate(args, model)


if __name__ == "__main__":
    main()
