from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *


class RelationNet(nn.Module):

    def __init__(self, concept_num, concept_dim, relation_num, relation_dim, sent_dim, concept_in_dim,
                 hidden_size, num_hidden_layers, num_attention_heads, fc_size, num_fc_layers, dropout,
                 pretrained_concept_emb=None, pretrained_relation_emb=None, freeze_ent_emb=True,
                 init_range=0, ablation=None, use_contextualized=False, emb_scale=1.0):

        super().__init__()
        self.init_range = init_range
        self.relation_num = relation_num
        self.ablation = ablation

        self.rel_emb = nn.Embedding(relation_num, relation_dim)
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                               scale=emb_scale)

        encoder_dim = {'no_qa': relation_dim, 'no_2hop_qa': relation_dim, 'no_rel': concept_dim * 2}.get(self.ablation, concept_dim * 2 + relation_dim)
        if self.ablation in ('encode_qas',):
            encoder_dim += sent_dim
        self.mlp = MLP(encoder_dim, hidden_size * 2, hidden_size,
                       num_hidden_layers, dropout, batch_norm=False, layer_norm=True)

        if ablation in ('multihead_pool',):
            self.attention = MultiheadAttPoolLayer(num_attention_heads, sent_dim, hidden_size)
        elif ablation in ('att_pool',):
            self.attention = AttPoolLayer(sent_dim, hidden_size)

        self.dropout_m = nn.Dropout(dropout)
        self.hid2out = MLP(hidden_size + sent_dim, fc_size, 1, num_fc_layers, dropout, batch_norm=False, layer_norm=True)
        self.activation = GELU()

        if self.init_range > 0:
            self.apply(self._init_weights)

        if pretrained_relation_emb is not None and ablation not in ('randomrel',):
            self.rel_emb.weight.data.copy_(pretrained_relation_emb)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, qa_ids, rel_ids, num_tuples, emb_data=None):
        """
        sent_vecs: tensor of shape (batch_size, d_sent)
        qa_ids: tensor of shape (batch_size, max_tuple_num, 2)
        rel_ids: tensor of shape (batch_size, max_tuple_num)
        num_tuples: tensor of shape (batch_size,)
        (emb_data: tensor of shape (batch_size, max_cpt_num, emb_dim))
        """

        bs, sl, _ = qa_ids.size()
        mask = torch.arange(sl, device=qa_ids.device) >= num_tuples.unsqueeze(1)
        if self.ablation in ('no_1hop', 'no_2hop', 'no_2hop_qa'):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            valid_mask = rel_ids > n_1hop_rel if self.ablation == 'no_1hop' else rel_ids <= n_1hop_rel
            mask = mask | ~valid_mask
        mask[mask.all(1), 0] = 0  # a temporary solution for instances that have no qar-pairs

        qa_emb = self.concept_emb(qa_ids.view(bs, -1), emb_data).view(bs, sl, -1)
        rel_embed = self.rel_emb(rel_ids)

        if self.ablation not in ('no_factor_mul',):
            n_1hop_rel = int(np.sqrt(self.relation_num))
            assert n_1hop_rel * (n_1hop_rel + 1) == self.relation_num
            rel_ids = rel_ids.view(bs * sl)
            twohop_mask = rel_ids >= n_1hop_rel
            twohop_rel = rel_ids[twohop_mask] - n_1hop_rel
            r1, r2 = twohop_rel // n_1hop_rel, twohop_rel % n_1hop_rel
            assert (r1 >= 0).all() and (r2 >= 0).all() and (r1 < n_1hop_rel).all() and (r2 < n_1hop_rel).all()
            rel_embed = rel_embed.view(bs * sl, -1)
            rel_embed[twohop_mask] = torch.mul(self.rel_emb(r1), self.rel_emb(r2))
            rel_embed = rel_embed.view(bs, sl, -1)

        if self.ablation in ('no_qa', 'no_rel', 'no_2hop_qa'):
            concat = rel_embed if self.ablation in ('no_qa', 'no_2hop_qa') else qa_emb
        else:
            concat = torch.cat((qa_emb, rel_embed), -1)

        if self.ablation in ('encode_qas',):
            sent_vecs_expanded = sent_vecs.unsqueeze(1).expand(bs, sl, -1)
            concat = torch.cat((concat, sent_vecs_expanded), -1)

        qars_vecs = self.mlp(concat)
        qars_vecs = self.activation(qars_vecs)

        if self.ablation in ('multihead_pool', 'att_pool'):
            pooled_vecs, att_scores = self.attention(sent_vecs, qars_vecs, mask)
        else:
            qars_vecs = qars_vecs.masked_fill(mask.unsqueeze(2).expand_as(qars_vecs), 0)
            pooled_vecs = qars_vecs.sum(1) / (~mask).float().sum(1).unsqueeze(1).float().to(qars_vecs.device)
            att_scores = None

        if self.ablation == 'no_kg':
            pooled_vecs[:] = 0

        logits = self.hid2out(self.dropout_m(torch.cat((pooled_vecs, sent_vecs), 1)))
        return logits, att_scores


class LMRelationNet(nn.Module):
    def __init__(self, model_name,
                 concept_num, concept_dim, relation_num, relation_dim, concept_in_dim, hidden_size, num_hidden_layers,
                 num_attention_heads, fc_size, num_fc_layers, dropout, pretrained_concept_emb=None,
                 pretrained_relation_emb=None, freeze_ent_emb=True, init_range=0, ablation=None,
                 use_contextualized=False, emb_scale=1.0, encoder_config={}):
        super().__init__()
        self.use_contextualized = use_contextualized
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = RelationNet(concept_num, concept_dim, relation_num, relation_dim, self.encoder.sent_dim, concept_in_dim,
                                   hidden_size, num_hidden_layers, num_attention_heads,
                                   fc_size, num_fc_layers, dropout, pretrained_concept_emb, pretrained_relation_emb,
                                   freeze_ent_emb=freeze_ent_emb, init_range=init_range, ablation=ablation,
                                   use_contextualized=use_contextualized, emb_scale=emb_scale)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        if self.use_contextualized:
            *lm_inputs, qa_ids, rel_ids, num_tuples, emb_data = inputs
        else:
            *lm_inputs, qa_ids, rel_ids, num_tuples = inputs
            emb_data = None
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs=sent_vecs, qa_ids=qa_ids, rel_ids=rel_ids, num_tuples=num_tuples, emb_data=emb_data)  # cxy-style param passing
        logits = logits.view(bs, nc)
        return logits, attn


class LMRelationNetDataLoader(object):

    def __init__(self, train_statement_path, train_rpath_jsonl,
                 dev_statement_path, dev_rpath_jsonl,
                 test_statement_path, test_rpath_jsonl,
                 batch_size, eval_batch_size, device, model_name,
                 max_tuple_num=200, max_seq_length=128,
                 is_inhouse=True, inhouse_train_qids_path=None, use_contextualized=False,
                 train_adj_path=None, train_node_features_path=None, dev_adj_path=None, dev_node_features_path=None,
                 test_adj_path=None, test_node_features_path=None, node_feature_type=None, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)

        num_choice = self.train_data[0].size(1)
        self.train_data += load_2hop_relational_paths(train_rpath_jsonl, train_adj_path,
                                                      emb_pk_path=train_node_features_path if use_contextualized else None,
                                                      max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
        self.dev_data += load_2hop_relational_paths(dev_rpath_jsonl, dev_adj_path,
                                                    emb_pk_path=dev_node_features_path if use_contextualized else None,
                                                    max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
            self.test_data += load_2hop_relational_paths(test_rpath_jsonl, test_adj_path,
                                                         emb_pk_path=test_node_features_path if use_contextualized else None,
                                                         max_tuple_num=max_tuple_num, num_choice=num_choice, node_feature_type=node_feature_type)
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)

        num_tuple_idx = -2 if use_contextualized else -1
        print('| train_num_tuples = {:.2f} | dev_num_tuples = {:.2f} | test_num_tuples = {:.2f} |'.format(self.train_data[num_tuple_idx].float().mean(),
                                                                                                          self.dev_data[num_tuple_idx].float().mean(),
                                                                                                          self.test_data[num_tuple_idx].float().mean() if test_statement_path else 0))

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

    def __getitem__(self, index):
        raise NotImplementedError()

    def get_node_feature_dim(self):
        return self.train_data[-1].size(-1) if self.use_contextualized else None

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
