# Multi-Hop Graph Relation Networks (EMNLP 2020)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo.png/800px-Pytorch_logo.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the repo of our EMNLP'20 [paper](https://arxiv.org/abs/2005.00646):

```
Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering
Yanlin Feng*, Xinyue Chen*, Bill Yuchen Lin, Peifeng Wang, Jun Yan and Xiang Ren.
EMNLP 2020.
*=equal contritbution
```

This repository also implements other graph encoding models for question answering (including vanilla LM finetuning).

- **RelationNet**
- **R-GCN**
- **KagNet** 
- **GConAttn**
- **KVMem**
- **MHGRN (or. MultiGRN)**

Each model supports the following text encoders:

- **LSTM**
- **GPT**
- **BERT** 
- **XLNet** 
- **RoBERTa**



## Resources

We provide preprocessed ConceptNet and pretrained entity embeddings for your own usage. These resources are independent of the source code.

***Note that the following reousrces can be download [here](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws).***

### ConceptNet (5.6.0)

| Description                  | Downloads                                                    | Notes                                                        |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Entity Vocab                 | [entity-vocab](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws) | one entity per line, space replaced by '_'                   |
| Relation Vocab               | [relation-vocab](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws) | one relation per line, merged                                |
| ConceptNet (CSV format)      | [conceptnet-5.6.0-csv](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws) | English tuples extracted from the full conceptnet with merged relations |
| ConceptNet (NetworkX format) | [conceptnet-5.6.0-networkx](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws) | NetworkX pickled format, pruned by filtering out stop words  |

### Entity Embeddings (Node Features)

Entity embeddings are packed into a matrix of shape (#ent, dim) and stored in numpy format. Use `np.load` to read the file. You may need to download the vocabulary files first.

| Embedding Model | Dimensionality | Description                                               | Downloads                                                    |
| --------------- | -------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| TransE          | 100            | Obtained using OpenKE with optim=sgd, lr=1e-3, epoch=1000 | [entities](<https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws>) [relations](<https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws>) |
| NumberBatch     | 300            | <https://github.com/commonsense/conceptnet-numberbatch>   | [entities](<https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws>) |
| BERT-based      | 1024           | Provided by Zhengwei                                      | [entities](https://drive.google.com/drive/folders/155codqEnsKazO8-BchF3rO_cP3EyYdws) |



## Dependencies

- [Python](<https://www.python.org/>) >= 3.6
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.1.0
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 2.0.0
- [tqdm](<https://github.com/tqdm/tqdm>)
- [dgl](<https://github.com/dmlc/dgl>) == 0.3.1 (GPU version)
- [networkx](<https://networkx.github.io/>) == 2.3

Run the following commands to create a conda environment (assume CUDA10):

```bash
conda create -n krqa python=3.6 numpy matplotlib ipython
source activate krqa
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
pip install dgl-cu100==0.3.1
pip install transformers==2.0.0 tqdm networkx==2.3 nltk spacy==2.1.6
python -m spacy download en
```



## Usage

### 1. Download Data

First, you need to download all the necessary data in order to train the model:

```bash
git clone https://github.com/INK-USC/MHGRN.git
cd MHGRN
bash scripts/download.sh
```

The script will:

- Download the [CommonsenseQA](<https://www.tau-nlp.org/commonsenseqa>) dataset
- Download [ConceptNet](<http://conceptnet.io/>)
- Download pretrained TransE embeddings

### 2. Preprocess

To preprocess the data, run:

```bash
python preprocess.py
```

By default, all available CPU cores will be used for multi-processing in order to speed up the process. Alternatively, you can use "-p" to specify the number of processes to use:

```bash
python preprocess.py -p 20
```

The script will:

- Convert the original datasets into .jsonl files (stored in `data/csqa/statement/`)
- Extract English relations from ConceptNet, merge the original 42 relation types into 17 types
- Identify all mentioned concepts in the questions and answers
- Extract subgraphs for each q-a pair

The preprocessing procedure takes approximately 3 hours on a 40-core CPU server. Most intermediate files are in .jsonl or .pk format and stored in various folders. The resulting file structure will look like:

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
        ├── statement/             (converted statements)
        ├── grounded/              (grounded entities)
        ├── paths/                 (unpruned/pruned paths)
        ├── graphs/                (extracted subgraphs)
        ├── ...
```

### 3. Hyperparameter Search (optional)

To search the parameters for RoBERTa-Large on CommonsenseQA:

```bash
bash scripts/param_search_lm.sh csqa roberta-large
```

To search the parameters for BERT+RelationNet on CommonsenseQA:

```bash
bash scripts/param_search_rn.sh csqa bert-large-uncased
```

### 4. Training 

Each graph encoding model is implemented in a single script:

| Graph Encoder                                                | Script      | Description                                                  |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| None                                                         | lm.py       | w/o knowledge graph                                          |
| [Relation Network](<https://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning.pdf>) | rn.py       |                                                              |
| [R-GCN](<https://arxiv.org/pdf/1703.06103.pdf>)              | rgcn.py     | Use `--gnn_layer_num ` and `--num_basis` to specify #layer and #basis |
| [KagNet](https://arxiv.org/abs/1909.02151)                   | kagnet.py   | Adapted from <https://github.com/INK-USC/KagNet>, still tuning |
| Gcon-Attn                                                    | gconattn.py |                                                              |
| KV-Memory                                                    | kvmem.py    |                                                              |
| MHGRN                                                        | grn.py      |                                                              |

Some important command line arguments are listed as follows (run `python {lm,rn,rgcn,...}.py -h` for a complete list):

| Arg                             | Values                                                     | Description                      | Notes                                                        |
| ------------------------------- | ---------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| `--mode`                        | {train, eval, ...}                                         | Training or Evaluation           | default=train                                                |
| `-enc, --encoder`               | {lstm, openai-gpt, bert-large-unased, roberta-large, ....} | Text Encoer                      | Model names (except for lstm) are the ones used by [huggingface-transformers](<https://github.com/huggingface/transformers>), default=bert-large-uncased |
| `--optim`                       | {adam, adamw, radam}                                       | Optimizer                        | default=radam                                                |
| `-ds, --dataset`                | {csqa, obqa}                                               | Dataset                          | default=csqa                                                 |
| `-ih, --inhouse`                | {0, 1}                                                     | Run In-house Split               | default=1, only applicable to CSQA                           |
| `--ent_emb`                     | {transe, numberbatch, tzw}                                 | Entity Embeddings                | default=tzw (BERT-based node features)                       |
| `-sl, --max_seq_len`            | {32, 64, 128, 256}                                         | Maximum Sequence Length          | Use 128 or 256 for datasets that contain long sentences! default=64 |
| `-elr, --encoder_lr`            | {1e-5, 2e-5, 3e-5, 6e-5, 1e-4}                             | Text Encoder LR                  | dataset specific and text encoder specific, default values in `utils/parser_utils.py` |
| `-dlr, --decoder_lr`            | {1e-4, 3e-4, 1e-3, 3e-3}                                   | Graph Encoder LR                 | dataset specific and model specific, default values in `{model}.py` |
| `--lr_schedule`                 | {fixed, warmup_linear, warmup_constant}                    | Learning Rate Schedule           | default=fixed                                                |
| `-me, --max_epochs_before_stop` | {2, 4, 6}                                                  | Early Stopping Patience          | default=2                                                    |
| `--unfreeze_epoch`              | {0, 3}                                                     | Freeze Text Encoder for N epochs | model specific                                               |
| `-bs, --batch_size`             | {16, 32, 64}                                               | Batch Size                       | default=32                                                   |
| `--save_dir`                    | str                                                        | Checkpoint Directory             | model specific                                               |
| `--seed`                        | {0, 1, 2, 3}                                               | Random Seed                      | default=0                                                    |

For example, run the following command to train a RoBERTa-Large model on CommonsenseQA:

```bash
python lm.py --encoder roberta-large --dataset csqa
```

To train a RelationNet with BERT-Large-Uncased as the encoder:

```bash
python rn.py --encoder bert-large-uncased
```

To **reproduce the reported results of MultiGRN** on CommonsenseQA official set:

```
bash scripts/run_grn_csqa.sh
```


### 5. Evaluation

To evaluate a trained model (you need to specify `--save_dir` if the checkpoint is not stored in the default directory):

```bash
python {lm,rn,rgcn,...}.py --mode eval [ --save_dir path/to/directory/ ]
```



## Use Your Own Dataset

- Convert your dataset to  `{train,dev,test}.statement.jsonl`  in .jsonl format (see `data/csqa/statement/train.statement.jsonl`)
- Create a directory in `data/{yourdataset}/` to store the .jsonl files
- Modify `preprocess.py` and perform subgraph extraction for your data
- Modify `utils/parser_utils.py` to support your own dataset
- Tune `encoder_lr`,`decoder_lr` and other important hyperparameters, modify `utils/parser_utils.py` and `{model}.py` to record the tuned hyperparameters
