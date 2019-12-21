import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.layers import *
from utils.data_utils import *


class GraphRelationLayer(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, input_size, hidden_size, output_size, sent_dim,
                 att_dim, att_layer_num, dropout=0.1, diag_decompose=False,
                 with_pairwise_scores=True, ablation=None, eps=1e-20, att_bias_init=-1e3, mlp_layer_norm=False):
        super().__init__()
        self.k = k
        self.n_head = n_head
        self.n_basis = n_basis
        self.hidden_size = hidden_size
        self.diag_decompose = diag_decompose
        self.n_type = n_type
        self.with_pairwise_scores = with_pairwise_scores
        self.ablation = ablation
        self.eps = eps

        assert input_size == output_size

        if diag_decompose and n_basis:
            raise ValueError('diag_decompose and n_basis > 0 cannot be true at the same time')

        self.U = nn.Parameter(torch.zeros(input_size, n_type * hidden_size))
        self.b = nn.Parameter(torch.zeros(n_type * hidden_size))
        nn.init.uniform_(self.U, -np.sqrt(6.0 / (input_size + hidden_size)), np.sqrt(6.0 / (input_size + hidden_size)))

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size, n_head))
            nn.init.uniform_(self.w_vs, -np.sqrt(6.0 / (hidden_size + hidden_size)), np.sqrt(6.0 / (hidden_size + hidden_size)))
            self.w_vs.data[1:] = 1
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size, hidden_size * n_head))
            nn.init.uniform_(self.w_vs, -np.sqrt(6.0 / (hidden_size + hidden_size)), np.sqrt(6.0 / (hidden_size + hidden_size)))
            self.w_vs.data[1:] = torch.eye(hidden_size, dtype=self.w_vs.dtype)[None, :, :, None].expand(k - 1, hidden_size, hidden_size,
                                                                                                        n_head).contiguous().view(k - 1, hidden_size, hidden_size * n_head)
        else:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size * hidden_size, n_basis))
            self.w_vs_co = nn.Parameter(torch.zeros(k, n_basis, n_head))
            nn.init.uniform_(self.w_vs, -np.sqrt(6.0 / (hidden_size + hidden_size)), np.sqrt(6.0 / (hidden_size + hidden_size)))
            nn.init.uniform_(self.w_vs_co, -np.sqrt(6.0 / (n_basis + n_head)), np.sqrt(6.0 / (n_basis + n_head)))
            self.w_vs.data[1:] = torch.eye(hidden_size, dtype=self.w_vs.dtype)[None, :, :, None].expand(k - 1, hidden_size, hidden_size,
                                                                                                        n_basis).contiguous().view(k - 1, hidden_size * hidden_size, n_basis)

        self.V1 = nn.Parameter(torch.zeros(hidden_size, output_size))
        self.V2 = nn.Parameter(torch.zeros(hidden_size, output_size))
        nn.init.uniform_(self.V1, -np.sqrt(6.0 / (hidden_size + output_size)), np.sqrt(6.0 / (hidden_size + output_size)))
        nn.init.uniform_(self.V2, -np.sqrt(6.0 / (hidden_size + output_size)), np.sqrt(6.0 / (hidden_size + output_size)))

        if 'no_att' not in self.ablation:
            self.start_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, 0.1, layer_norm=mlp_layer_norm, init_last_layer_bias_to_zero=True)
            self.end_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, 0.1, layer_norm=mlp_layer_norm, init_last_layer_bias_to_zero=True)
            self.path_uni_attention = MLP(sent_dim, att_dim, n_head * k, att_layer_num, 0.1, layer_norm=mlp_layer_norm, init_last_layer_bias_to_zero=True)
            bias_init = torch.zeros((n_head, k), dtype=torch.float32)
            bias_init[:-1, 1:] = att_bias_init
            self.path_uni_attention.layers[-1].bias.data.copy_(bias_init.view(-1))

            if 'no_trans' not in self.ablation:
                if self.with_pairwise_scores:
                    self.path_pair_attention = MLP(sent_dim, att_dim, n_head ** 2, 1, 0.1, layer_norm=mlp_layer_norm, init_last_layer_bias_to_zero=True)
                self.trans_scores = nn.Parameter(torch.zeros(n_head ** 2))

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, S, H, A, node_type):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """

        k, h_size, n_head, n_basis = self.k, self.hidden_size, self.n_head, self.n_basis
        bs, n_node, i_size = H.size()

        node_type = node_type.view(-1)

        X = H.view(bs * n_node, i_size)
        X = (X.mm(self.U) + self.b).view(bs * n_node, -1, h_size)
        X = X[torch.arange(bs * n_node, device=X.device), node_type]
        X = X.view(bs, n_node, h_size)

        if self.diag_decompose:
            W = self.w_vs
            if 'mask_2hop' in self.ablation:
                W = W.clone()
                W[1:] = 1
        elif n_basis == 0:
            W = self.w_vs
            if 'mask_2hop' in self.ablation:
                raise NotImplementedError()
        else:
            W = self.w_vs.bmm(self.w_vs_co).view(k, h_size, h_size * n_head)
            if 'mask_2hop' in self.ablation:
                raise NotImplementedError()

        if 'no_att' not in self.ablation:
            bi = torch.arange(bs).unsqueeze(-1).expand(bs, n_node).contiguous().view(-1)
            start_attn = self.start_attention(S)
            start_attn = torch.exp(start_attn - start_attn.max(1, keepdim=True)[0])  # softmax trick to avoid numeric overflow
            start_attn = start_attn[bi, node_type].view(bs, n_node)

            end_attn = self.end_attention(S)
            end_attn = torch.exp(end_attn - end_attn.max(1, keepdim=True)[0])
            end_attn = end_attn[bi, node_type].view(bs, n_node)

            uni_attn = self.path_uni_attention(S).view(bs, n_head, k)  # (bs, n_head, k)
            if 'mask_2hop' in self.ablation:
                uni_attn_mask = uni_attn.new_zeros((bs, n_head, k), dtype=torch.uint8)
                uni_attn_mask[:, :-1, 1:] = 1
                uni_attn = uni_attn.masked_fill(uni_attn_mask, -np.inf)
            uni_attn = torch.exp(uni_attn - uni_attn.max(1, keepdim=True)[0]).view(bs * n_head, k)

            if 'no_trans' not in self.ablation:
                if self.with_pairwise_scores:
                    T = self.path_pair_attention(S) + self.trans_scores
                else:
                    T = self.trans_scores.unsqueeze(0).expand(bs, n_head ** 2)
                T = torch.exp(T - T.max(1, keepdim=True)[0])
                T = T.view(bs, n_head, n_head)
        else:
            start_attn = torch.ones((bs, n_node), device=X.device)
            end_attn = torch.ones((bs, n_node), device=X.device)
            uni_attn = torch.ones((bs * n_head, k), device=X.device)
            T = torch.ones((bs, n_head, n_head), device=X.device)

        A = A.view(bs * n_head, n_node, n_node)

        if 'no_trans' in self.ablation or 'no_att' in self.ablation:
            Z = X * start_attn.unsqueeze(2)
            for t in range(k):
                if self.diag_decompose:
                    Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_node, h_size, n_head)
                else:
                    Z = Z.matmul(W[t]).view(bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, t, None, None]
                Z = A.bmm(Z)
                Z = Z.view(bs, n_head, n_node, h_size).sum(1)
            Z = Z * end_attn.unsqueeze(2)

            D = start_attn.clone()
            for t in range(k):
                D = D.repeat(1, n_head).view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, t, None, None]
                D = A.bmm(D)
                D = D.view(bs, n_head, n_node).sum(1)
            D = D * end_attn

        else:
            Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
            for t in range(k):
                # there is another implementation that possibly consumes less memory if n_node > h_size
                # https://github.com/Evan-Feng/knowledge-reasoning-qaq/blob/26ddd8809b9a62867289b360a3f0f3eecddb5330/khop_rgcn.py
                if t == 0:
                    if self.diag_decompose:
                        Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_node, h_size, n_head)
                    else:
                        Z = Z.matmul(W[t]).view(bs, n_node, h_size, n_head)
                else:
                    if self.diag_decompose:
                        Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_head, n_node, h_size, n_head)
                    else:
                        Z = Z.matmul(W[t]).view(bs, n_head, n_node, h_size, n_head)
                    Z = Z * T[:, :, None, None, :]
                    Z = Z.sum(1)  # (bs, n_node, h_size,n_head)

                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, t, None, None]
                Z = A.bmm(Z)
                Z = Z.view(bs, n_head, n_node, h_size)
            if k >= 1:
                Z = Z.sum(1)
            Z = Z * end_attn.unsqueeze(2)

            # compute the normalization factor
            D = start_attn
            for t in range(k):
                if t == 0:
                    D = D.unsqueeze(1).expand(bs, n_head, n_node)
                else:
                    D = D.unsqueeze(2) * T.unsqueeze(3)
                    D = D.sum(1)
                D = D.contiguous().view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, t, None, None]
                D = A.bmm(D)
                D = D.view(bs, n_head, n_node)
            if k >= 1:
                D = D.sum(1)
            D = D * end_attn  # (bs, n_node)

        Z = Z / (D.unsqueeze(2) + self.eps)

        if 'early_relu' in self.ablation:
            output = X.matmul(self.V1) + self.activation(Z.matmul(self.V2))
        else:
            output = self.activation(X.matmul(self.V1) + Z.matmul(self.V2))
        output = self.dropout(output)
        return output


class GraphRelationEncoder(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, n_layer, input_size, hidden_size, sent_dim, att_dim, att_layer_num, dropout, diag_decompose,
                 with_pairwise_scores=True, ablation=None, eps=1e-20, att_bias_init=-1e3, mlp_layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([GraphRelationLayer(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis,
                                                        input_size=input_size, hidden_size=hidden_size, output_size=input_size, sent_dim=sent_dim,
                                                        att_dim=att_dim, att_layer_num=att_layer_num,
                                                        dropout=dropout, diag_decompose=diag_decompose,
                                                        with_pairwise_scores=with_pairwise_scores,
                                                        ablation=ablation, eps=eps, att_bias_init=att_bias_init,
                                                        mlp_layer_norm=mlp_layer_norm) for _ in range(n_layer)])

    def forward(self, S, H, A, node_type_ids):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type_ids: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """
        for layer in self.layers:
            H = layer(S, H, A, node_type_ids)
        return H


class GraphRelationNet(nn.Module):
    def __init__(self, k, n_type, n_basis, n_layer, sent_dim, diag_decompose,
                 n_concept, n_relation, concept_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc, p_concat,
                 pretrained_concept_emb=None, ablation=None,
                 with_pairwise_scores=True, init_range=0.0, eps=1e-20, att_bias_init=-1e-3, mlp_layer_norm=False, fc_layer_norm=False):
        super().__init__()
        self.ablation = ablation
        self.init_range = init_range

        self.concept_emb = nn.Embedding(n_concept, concept_dim)
        self.gnn = GraphRelationEncoder(k=k, n_type=n_type, n_head=n_relation, n_basis=n_basis, n_layer=n_layer,
                                        input_size=concept_dim, hidden_size=(concept_dim * 2), sent_dim=sent_dim,
                                        att_dim=att_dim, att_layer_num=att_layer_num, dropout=p_gnn,
                                        diag_decompose=diag_decompose, with_pairwise_scores=with_pairwise_scores,
                                        ablation=ablation, eps=eps, att_bias_init=att_bias_init, mlp_layer_norm=mlp_layer_norm)
        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)
        self.fc = MLP(d_v * n_attention_head + sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=fc_layer_norm)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_concat = nn.Dropout(p_concat)

        if init_range > 0:
            self.apply(self._init_weights)

        if pretrained_concept_emb is not None:
            self.concept_emb.weight.data.copy_(pretrained_concept_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, GraphRelationLayer):
            for param_name in ('U', 'w_vs', 'V1', 'V2'):
                getattr(module, param_name).data.normal_(mean=0.0, std=self.init_range)
            for param_name in ('b', 'trans_scores'):
                if hasattr(module, param_name):
                    getattr(module, param_name).data.zero_()
            if hasattr(module, 'w_vs_co'):
                getattr(module, 'w_vs_co').data.fill_(1.0)

    def forward(self, sent_vecs, concept_ids, node_type_ids, adj_lengths, adj):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node

        returns: (batch_size, 1)
        """
        gnn_input = self.dropout_e(self.concept_emb(concept_ids))
        gnn_output = self.gnn(sent_vecs, gnn_input, adj, node_type_ids)
        adj_lengths = torch.max(adj_lengths, adj_lengths.new_ones(()))  # a temporary solution to avoid zero node
        mask = torch.arange(concept_ids.size(1), device=adj.device) >= adj_lengths.unsqueeze(1)
        graph_vecs, pool_attn = self.pooler(sent_vecs, gnn_output, mask)
        concat = self.dropout_concat(torch.cat((graph_vecs, sent_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn


class LMGraphRelationNet(nn.Module):
    def __init__(self, model_name, k, n_type, n_basis, n_layer, diag_decompose,
                 n_concept, n_relation, concept_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc, p_concat,
                 pretrained_concept_emb=None, ablation=None,
                 with_pairwise_scores=True, init_range=0.0, eps=1e-20, att_bias_init=-1e-3,
                 mlp_layer_norm=False, fc_layer_norm=False, encoder_config={}):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = GraphRelationNet(k, n_type, n_basis, n_layer, self.encoder.sent_dim, diag_decompose,
                                        n_concept, n_relation, concept_dim, n_attention_head,
                                        fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc, p_concat,
                                        pretrained_concept_emb=pretrained_concept_emb, ablation=ablation,
                                        with_pairwise_scores=with_pairwise_scores, init_range=init_range, eps=eps,
                                        att_bias_init=att_bias_init, mlp_layer_norm=mlp_layer_norm, fc_layer_norm=fc_layer_norm)

    def forward(self, *inputs, layer_id=-1):
        """
        sent_vecs: (batch_size, num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)
        adj: (batch_size, num_choice, n_head, n_node, n_node)
        adj_lengths: (batch_size, num_choice)
        node_type_ids: (batch_size, num_choice n_node)

        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        *lm_inputs, concept_ids, node_type_ids, adj_lengths, adj = inputs
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.to(concept_ids.device), concept_ids, node_type_ids, adj_lengths, adj)
        logits = logits.view(bs, nc)
        return logits, attn


class LMGraphRelationNetDataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
        *self.train_decoder_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, num_choice)
        *self.dev_decoder_data, self.dev_adj_data, n_rel = load_adj_data(dev_adj_path, max_node_num, num_choice)
        assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        # pre-allocate an empty batch adj matrix
        self.adj_empty = torch.zeros((self.batch_size, num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32)
        self.eval_adj_empty = torch.zeros((self.eval_batch_size, num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data, n_rel = load_adj_data(test_adj_path, max_node_num, num_choice)
            assert all(len(self.test_qids) == len(self.test_adj_data) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

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
        return MultiGPUAdjDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.adj_empty, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                             tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                                 tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)
        else:
            return MultiGPUAdjDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
                                                 tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_empty=self.eval_adj_empty, adj_data=self.test_adj_data)
