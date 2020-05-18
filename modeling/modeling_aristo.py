from transformers import RobertaConfig, RobertaModel
from transformers.modeling_roberta import RobertaClassificationHead
import torch
import torch.nn as nn


class AristoForMultipleChoice(nn.Module):

    def __init__(self, dropout=0.1, from_checkpoint=None):
        super().__init__()
        self.encoder = AristoEncoder(from_checkpoint=from_checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(self.encoder.sent_dim, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        sent_vecs, all_hidden_states = self.encoder(*inputs, layer_id=layer_id)
        sent_vecs = self.dropout(sent_vecs)
        logits = self.decoder(sent_vecs).view(bs, nc)
        return logits


class AristoEncoder(nn.Module):
    def __init__(self, from_checkpoint=None):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-large', output_hidden_states=True)
        self.roberta = RobertaModel(config=config)
        if from_checkpoint is not None:
            self.roberta = RobertaModel.from_pretrained(from_checkpoint, config=config)
        self.sent_dim = self.roberta.config.hidden_size
        # self._debug = 1
        self._debug = -1

    def forward(self, *inputs, layer_id=-1):
        # pull the last layer of roberta outputs
        input_ids, attention_mask = inputs
        if self._debug > -1:
            self._debug -= 1

        if self._debug == 0:
            print(f"attention_mask = {attention_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")

        # Segment ids are not used by RoBERTa

        # TODO: token_type_ids is commented!!
        outputs = self.roberta(input_ids=input_ids,
                               # token_type_ids=util.combine_initial_dims(segment_ids),
                               attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        sent_vecs = self.roberta.pooler(hidden_states)

        if self._debug == 0:
            print(f"sent_vecs = {sent_vecs}")

        return sent_vecs, all_hidden_states


class Aristo(nn.Module):
    def __init__(self):
        super().__init__()
        config = RobertaConfig.from_pretrained('roberta-large', output_hidden_states=True)
        self.roberta = RobertaModel(config=config)
        config.num_labels = 1
        self.classifier = RobertaClassificationHead(config=config)
        # self._debug = 1
        self._debug = -1

    def forward(self, *inputs):
        # pull the last layer of roberta outputs
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]
        input_ids, attention_mask = inputs
        if self._debug > -1:
            self._debug -= 1

        if self._debug == 0:
            print(f"attention_mask = {attention_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            # print(f"token_type_ids = {token_type_ids}")

        # Segment ids are not used by RoBERTa

        # TODO: token_type_ids is not used!!
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        cls_output = outputs[0]  # the last layer
        logits = self.classifier(cls_output).view(bs, nc)
        return logits

