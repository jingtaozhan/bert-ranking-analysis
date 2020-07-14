# bert-ranking-analysis

First, make sure you already installed [🤗 Transformers](https://github.com/huggingface/transformers):

```bash
pip install transformers
git clone https://github.com/jingtaozhan/RepBERT-Index
cd RepBERT-Index
```

Next, download `collectionandqueries.tar.gz` from [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking). It contains passages, queries, and qrels.

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