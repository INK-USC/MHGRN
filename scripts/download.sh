# download ConceptNet
mkdir -p data/
mkdir -p data/cpnet/
wget -nc -P data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../../

# download GloVe vectors
mkdir -p data/glove/
wget -nc -P data/glove/ http://nlp.stanford.edu/data/glove.6B.zip
yes n | unzip data/glove/glove.6B.zip -d data/glove/

# download TransE embeddings
mkdir -p data/transe/
wget -nc -P data/transe/ https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.ent.npy
wget -nc -P data/transe/ https://csr.s3-us-west-1.amazonaws.com/glove.transe.sgd.rel.npy

# download numberbatch embeddings
wget -nc -P data/transe https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
cd data/transe
yes n | gzip -d numberbatch-en-19.08.txt.gz
cd ../../

# download CommensenseQA dataset
mkdir -p data/csqa/
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget -nc -P data/csqa/ https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl

# download BERT features
mkdir -p data/csqa/bert/
wget -nc -P data/csqa/bert/ https://csr.s3-us-west-1.amazonaws.com/train.bert.large.layer-2.epoch1.npy
wget -nc -P data/csqa/bert/ https://csr.s3-us-west-1.amazonaws.com/dev.bert.large.layer-2.epoch1.npy

# download fairseq data
mkdir -p data/csqa/fairseq/
mkdir -p data/csqa/fairseq/official/
mkdir -p data/csqa/fairseq/inhouse/
wget -nc -O data/csqa/fairseq/official/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

# create output folders
mkdir -p data/csqa/grounded/
mkdir -p data/csqa/paths/
mkdir -p data/csqa/graph/
mkdir -p data/csqa/statement/
mkdir -p data/csqa/tokenized/
mkdir -p data/csqa/roberta/

# download SciTail dataset
wget -nc -P data/scitail/ http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip
yes n | unzip data/scitail/SciTailV1.1.zip -d data/scitail/

# create output folders
mkdir -p data/scitail/fairseq/
mkdir -p data/scitail/fairseq/official/
mkdir -p data/scitail/fairseq/inhouse/
wget -nc -O data/scitail/fairseq/official/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
mkdir -p data/scitail/grounded/
mkdir -p data/scitail/paths/
mkdir -p data/scitail/graph/
mkdir -p data/scitail/statement/
mkdir -p data/scitail/tokenized/
mkdir -p data/scitail/roberta/

# download SocialIQA dataset
wget -nc -P data/socialiqa/ https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip
yes n | unzip data/socialiqa/socialiqa-train-dev.zip -d data/socialiqa/

# create output folders
mkdir -p data/socialiqa/fairseq/
mkdir -p data/socialiqa/fairseq/official/
mkdir -p data/socialiqa/fairseq/inhouse/
wget -nc -O data/socialiqa/fairseq/official/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
mkdir -p data/socialiqa/grounded/
mkdir -p data/socialiqa/paths/
mkdir -p data/socialiqa/graph/
mkdir -p data/socialiqa/statement/
mkdir -p data/socialiqa/tokenized/
mkdir -p data/socialiqa/roberta/

# download OpenBookQA dataset
wget -nc -P data/obqa/ https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip
yes n | unzip data/obqa/OpenBookQA-V1-Sep2018.zip -d data/obqa/

# create output folders
mkdir -p data/obqa/fairseq/
mkdir -p data/obqa/fairseq/official/
mkdir -p data/obqa/fairseq/inhouse/
wget -nc -O data/obqa/fairseq/official/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
mkdir -p data/obqa/grounded/
mkdir -p data/obqa/paths/
mkdir -p data/obqa/graph/
mkdir -p data/obqa/statement/
mkdir -p data/obqa/tokenized/
mkdir -p data/obqa/roberta/
