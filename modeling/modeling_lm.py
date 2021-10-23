import torch
import torch.nn as nn

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import BatchGenerator, load_input_tensors


class LMForMultipleChoice(nn.Module):

    def __init__(self, model_name, dropout=0.1, from_checkpoint=None, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, from_checkpoint=from_checkpoint, **encoder_config)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(self.encoder.sent_dim, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        sent_vecs, all_hidden_states = self.encoder(*inputs, layer_id=layer_id)
        sent_vecs = self.dropout(sent_vecs)
        logits = self.decoder(sent_vecs).view(bs, nc)
        return logits


class LMDataLoader(object):

    def __init__(self, train_statement_path, dev_statement_path, test_statement_path,
                 batch_size, eval_batch_size, device, model_name, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None, subsample=1.0, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)
        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_data = [x[:n_train] for x in self.train_data]
                assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def train_eval(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors=self.train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
