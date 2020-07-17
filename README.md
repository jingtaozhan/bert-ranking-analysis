# An Analysis of BERT in Document Ranking

To increase the explainability of the ranking process performed by BERT, we investigate a state-of-the-art [BERT-based ranking model](https://arxiv.org/abs/1901.04085) with focus on its attention mechanism and interaction behavior. 
Firstly, we look into the evolving of the attention distribution. It shows that in each step, BERT dumps redundant attention weights on tokens with high document frequency (such as periods). This may lead to a potential threat to the model robustness and should be considered in future studies. 
Secondly, we study how BERT models interactions between query and document and find that BERT aggregates document information to query token representations through their interactions, but extracts query-independent representations for document tokens. It indicates that it is possible to transform BERT into a more efficient representation-focused model. 

These findings help us better understand the ranking process by BERT and may inspire future improvement. For more details, check out our paper:
+ Zhan et al.  [An Analysis of BERT in Document Ranking](http://www.thuir.cn/group/~YQLiu/publications/SIGIR2020Zhan.pdf)

In the following, we present instructions on how to replicate our experimental results.

## Prepare
Our implementation is based on Pytorch. Make sure you already installed [🤗 Transformers](https://github.com/huggingface/transformers):

```bash
pip install transformers
git clone https://github.com/jingtaozhan/bert-ranking-analysis
cd bert-ranking-analysis
```

## Data Process
Download `collectionandqueries.tar.gz` from [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking). It contains passages, queries, and qrels.

```bash
mkdir data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
mkdir msmarco-passage
tar xvfz collectionandqueries.tar.gz -C msmarco-passage
```

To confirm, `collectionandqueries.tar.gz` should have MD5 checksum of `31644046b18952c1386cd4564ba2ae69`.

To reduce duplication of effort in training and testing, we tokenize queries and passages in advance. This should take some time (about 3-4 hours). Besides, we convert tokenized passages to numpy memmap array, which can greatly reduce the memory overhead at run time.

```bash
python convert_text_to_tokenized.py --tokenize_queries --tokenize_collection
python convert_collection_to_memmap.py
```

We adopt the [BERT_Base_trained_on_MSMARCO.zip](https://drive.google.com/file/d/1cyUrhs7JaCJTTu-DjFUqP6Bs4f8a6JTX/view) model provided by [Passage Re-ranking with BERT](https://github.com/nyu-dl/dl4marco-bert). Please download and unzip to directory `./data/BERT_Base_trained_on_MSMARCO`.

## Attention Pattern
We firstly save the attention map to disk and then draw the attention distribution. We find no significant difference in attention distribution between relevant query-passage pairs and irrelevant ones. In this public implemantation, `attpattern.save_att` only considers the relevant ones.

```bash
python -m attpattern.save_att
python -m attpattern.draw
```

The attention map is saved at `./data/attention/` and the figure is saved at `./data/avg_att.pdf`.

## Attribution
`attribution.run_attr` uses multiprocess to compute attribution for different layers. It saves attributions to `./data/attribution`. `attpattern.draw` plots the figure and saves to `./data/attribution.pdf`.

```bash
python -m attribution.run_attr --attr_segment query --gpus 0 1 2 3
python -m attribution.run_attr --attr_segment para --gpus 0 1 2 3
python -m attpattern.draw
```

## Probing
We use `probing.sample_traindata` to sample training query-passage pairs from MSMARCO training set. It is provided at `./data/sample.train.tsv` so you do not need run this script. 

To train probing classifiers, we use `probing.save_embed` to computes and saves the intermediate representations in the training set and use `probing.runprob` to train. As for the evaluation, we use `probing.save_embed` to computes and saves the intermediate representations in the dev set. We call `probing.runprob` to load the trained probing classifiers to predict.

Take periods for example, you can run the following commad:

```bash
python -m probing.save_embed --keys periods_in_passage --rank_file ./data/sample.train.tsv \
       --mode train --gpu 0 % It will save period representations of all layers
python -m probing.runprob --key periods_in_passage --gpu 0 --do_train --layer 0
python -m probing.runprob --key periods_in_passage --gpu 1 --do_train --layer 1
... ...
python -m probing.runprob --key periods_in_passage --gpu 2 --do_train --layer 11
% The training data consumes much storage space, you may consider deleting them at ./data/probing/embed/train
% The trained models are saved at ./data/probing/models

python -m attpattern.draw
python -m probing.save_embed --keys periods_in_passage --rank_file ./data/anserini.dev.small.top100.tsv \
       --mode dev.small --gpu 0 % It will save period representations of all layers
python -m probing.runprob --key periods_in_passage --gpu 0 --do_eval --layer 0
python -m probing.runprob --key periods_in_passage --gpu 1 --do_eval --layer 1
... ...
python -m probing.runprob --key periods_in_passage --gpu 2 --do_eval --layer 11
% The dev data consumes much storage space, you may consider deleting them at ./data/probing/embed/dev.small
% The output evaluate results are saved at ./data/probing/eval
```

It is quite boring to run so many commands, thus you can directly run a wrapped script as follows:

```bash
python -m attribution.multirun --save_train_embed --do_train --save_eval_embed --do_eval \
       --keys periods_in_passage --gpus 0 1 2 3 --layers 0 1 2 3 4 5 6 7 8 9 10 11
% For [CLS] tokens, the --layers argument should be set to 0 1 2 3 4 5 6 7 8 9 10 11 12
```

We also provide our [trained probing classifiers](https://drive.google.com/file/d/1LN8uRk2t8T8SwfrykRneLnDXRJjBMVu_/view?usp=sharing). You can download it and unzip to `./data/probing/models`. Thus, you do not need to save the representations of training set nor train the probing classifiers. The command should be:

```bash
python -m attribution.multirun --save_eval_embed --do_eval \
       --keys periods_in_passage --gpus 0 1 2 3 --layers 0 1 2 3 4 5 6 7 8 9 10 11
% For [CLS] tokens, the --layers argument should be set to 0 1 2 3 4 5 6 7 8 9 10 11 12
```

Once you finish running all kinds of tokens, namely `cls`, `seps`, `periods_in_passage`, `all_query_tokens`, 
        `rand_passage_tokens`, `stopwords_in_passage`, `query_tokens_in_passage`, you can plot the figure:
```bash
python -m attribution.draw
```
The figure is saved to `./data/probing.pdf`

## Mask

There are three mask methods and 12 layers. You need to combine them and run the following commands:

```bash
python -m mask.bert_forward --mask_target mask_both_query_para --mask_layer_num 0 --gpu 0
python -m mask.bert_forward --mask_target mask_both_query_para --mask_layer_num 1 --gpu 1
...
python -m mask.bert_forward --mask_target mask_both_query_para --mask_layer_num 11 --gpu 2
python -m mask.bert_forward --mask_target mask_para_from_query --mask_layer_num 0 --gpu 3
...
```

You may prefer a wrapped script instead:

```bash
python -m mask.multi_process_run --gpus 0 1 2 3  \
       --mask_targets mask_both_query_para mask_para_from_query mask_query_from_para \
       --mask_layer_nums 0 1 2 3 4 5 6 7 8 9 11
```

To plot the figures, run the following command:

```bash
python -m mask.draw
```

The figure is saved to `./data/mask.pdf`.