# BERT-GCN-NLI
Graph construction and BERT-GCN for NLI task

## Resources

### Statement vectors extracted from finetuned LM

|   Model    | Setting  | Dev Acc | Layer | Epoch |                                                              |                                                              |                                                              |
| :--------: | :------: | :-----: | :---: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| BERT-Large | Official |  63.14  |  -2   |   1   | [train](<https://csr.s3-us-west-1.amazonaws.com/train.bert.large.layer-2.epoch1.npy>) | [dev](<https://csr.s3-us-west-1.amazonaws.com/dev.bert.large.layer-2.epoch1.npy>) | [test](<https://csr.s3-us-west-1.amazonaws.com/test.bert.large.layer-2.epoch1.npy>) |
| BERT-Large | In-house |         |       |       |                                                              |                                                              |                                                              |
|  RoBERTa   | Official |         |       |       |                                                              |                                                              |                                                              |
|  RoBERTa   | In-house |         |       |       |                                                              |                                                              |                                                              |

### ConceptNet embeddings

| Model  | Dimensionality | Optimizer | Learning rate | Initialization | Epoch |                                                              |                                                              |
| :----: | :------------: | :-------: | :-----------: | :------------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| TransE |      100       |    SGD    |     0.001     | GloVe-MaxPool  | 1000  | [entities](<https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.ent.npy>) | [relations](<https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.rel.npy>) |
| TransE |      200       |    SGD    |     0.001     | GloVe-MaxPool  | 1000  | [entities](<https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.lr0.001.d200.e1000.ent.npy>) | [relations](<https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.lr0.001.d200.e1000.rel.npy>) |
| TransE |      300       |           |               |                |       |                                                              |                                                              |
| TransE |      600       |           |               |                |       |                                                              |                                                              |

## Dependencies

- [Python](<https://www.python.org/>) >= 3.6
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.1
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 2.0.0
- [tqdm](<https://github.com/tqdm/tqdm>)
- [dgl](<https://github.com/dmlc/dgl>) == 0.3 (GPU version)
- [networkx](<https://networkx.github.io/>) == 2.3
- [fairseq](<https://github.com/pytorch/fairseq>) == 0.8.0

Run the following commands to create a conda environment (assume CUDA10):

```bash
conda create -n krqa python=3.6 numpy matplotlib ipython
source activate krqa
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install dgl-cu100
pip install transformers==2.0.0 tqdm networkx==2.3 nltk spacy==2.1.6
python -m spacy download en
```

## Usage

### 1. Download Data

First, you need to download all the necessary data in order to train the model:

```bash
git clone https://github.com/shanzhenren/BERT-GCN-NLI.git
cd BERT-GCN-NLI
bash scripts/download.sh
```

The script will:

- Download the [CommonsenseQA](<https://www.tau-nlp.org/commonsenseqa>) dataset
- Download [ConceptNet](<http://conceptnet.io/>)
- Download pretrained [GloVe](<https://nlp.stanford.edu/projects/glove/>) embeddings
- Download pretrained TransE embeddings
- Download extracted BERT-large features

### 2. Preprocess

To preprocess the data, run:

```bash
python preprocess.py
```

By default, all available CPU cores will be used for multi-processing in order to speed up the process. Alternatively, you can use "-p" to specify the number of processes to use:

```bash
python preprocess.py -p 20
```
To preprocess a particular dataset, for example, CommonsenseQA, run:

```bash
python preprocess.py --run common csqa
```

The script will:

- Convert the original datasets into entailment datasets
- Extract English relations from ConceptNet, merge the original 42 relation types into 17 types
- Find all mentioned concepts in the questions and answers
- Generate paths for each question-answer paths and perform path pruning
- Generate the final grounded schema graphs

The preprocessing procedure of CommonsenseQA takes approximately 3 hours on a 40-core CPU server. All intermediate files are in .jsonl or .pk format and stored in various folders. The resulting file structure will look like:

```plain
.
├── README.md
└── data/
    ├── cpnet/                 (prerocessed ConceptNet)
    ├── glove/                 (pretrained GloVe embeddings)
    ├── transe/                (pretrained TransE embeddings)
    └── csqa/
        ├── train_rand_split.jsonl
        ├── dev_rand_split.jsonl
        ├── test_rand_split_no_answers.jsonl
        ├── bert/                  (features extracted from BERT-large)
        ├── statement/             (converted statements)
        ├── tokenized/             (tokenized statements)
        ├── grounded/              (grounded concepts)
        ├── paths/                 (unpruned/pruned paths)
        └── graphs/                (schema graphs)
```

### 3. Training 

Run the following command to train a GCN+BERT model:

```bash
python main.py --model gcn
```

