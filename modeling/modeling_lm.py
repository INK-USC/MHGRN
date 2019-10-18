import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (OpenAIGPTConfig, OpenAIGPTModel, OpenAIGPTTokenizer, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          BertConfig, BertModel, BertTokenizer, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNetConfig, XLNetModel, XLNetTokenizer, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          RobertaConfig, RobertaModel, RobertaTokenizer, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
from transformers import AutoModel
import numpy as np
from utils.data_utils import BatchGenerator, load_input_tensors, get_gpt_token_num

MODEL_CLASSES = {
    'gpt': (OpenAIGPTConfig, OpenAIGPTModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


class LMEncoder(nn.Module):
    valid_model_types = set(MODEL_CLASSES.keys())

    def __init__(self, model_name, output_token_states=False):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta',)

        self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        if self.model_type in ('gpt',):
            self.lm.resize_token_embeddings(get_gpt_token_num())

        self.sent_dim = self.lm.config.n_embd if self.model_type in ('gpt',) else self.lm.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''

        if self.model_type in ('gpt',):  # gpt
            input_ids, cls_token_ids, lm_labels = inputs  # lm_labels is not used
            outputs = self.lm(input_ids)
        else:  # bert / xlnet / roberta
            input_ids, attention_mask, token_type_ids, output_mask = inputs
            outputs = self.lm(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        all_hidden_states = outputs[-1]
        assert (outputs[0] == all_hidden_states[-1]).all()
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('gpt',):
            cls_token_ids = cls_token_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            sent_vecs = hidden_states.gather(1, cls_token_ids).squeeze(1)
        elif self.model_type in ('xlnet',):
            sent_vecs = hidden_states[:, -1]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            sent_vecs = self.lm.pooler(hidden_states)
        return sent_vecs, all_hidden_states


class LMForMultipleChoice(nn.Module):

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = LMEncoder(model_name)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(self.encoder.sent_dim, 1)

    def forward(self, *inputs):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        sent_vecs, all_hidden_states = self.encoder(*inputs)
        sent_vecs = self.dropout(sent_vecs)
        logits = self.decoder(sent_vecs).view(bs, nc)
        return logits


class LMDataLoader(object):

    def __init__(self, train_statement_path, dev_statement_path, test_statement_path,
                 batch_size, eval_batch_size, device, model_name, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)
        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

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
