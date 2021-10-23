from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *


class RGCNLayer(nn.Module):

    def __init__(self, n_head, n_basis, input_size, output_size, dropout=0.1, diag_decompose=False):
        super().__init__()
        self.n_head = n_head
        self.n_basis = n_basis
        self.output_size = output_size
        self.diag_decompose = diag_decompose

        assert input_size == output_size

        if diag_decompose and (input_size != output_size):
            raise ValueError('If diag_decompose=True then input size must equaul to output size')
        if diag_decompose and n_basis:
            raise ValueError('diag_decompose and n_basis > 0 cannot be true at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(input_size, n_head))
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(input_size, output_size * n_head))
        else:
            self.w_vs = nn.Parameter(torch.zeros(input_size, output_size, n_basis))
            self.w_vs_co = nn.Parameter(torch.zeros(n_basis, n_head))
            nn.init.xavier_uniform_(self.w_vs_co)
        nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (input_size + output_size)))

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, normalized_adj_t):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        normalized_adj_t: tensor of shape (b_sz*n_head, n_node, n_node)
            normalized_adj_t[:, j, i] ==  1/n indicates a directed edge i --> j and in_degree(j) == n
        """

        o_size, n_head, n_basis = self.output_size, self.n_head, self.n_basis
        bs, n_node, _ = inputs.size()

        if self.diag_decompose:
            output = (inputs.unsqueeze(-1) * self.w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size
        elif n_basis == 0:
            w_vs = self.w_vs
            output = inputs.matmul(w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size
        else:
            w_vs = self.w_vs.matmul(self.w_vs_co).view(-1, o_size * n_head)
            output = inputs.matmul(w_vs).view(bs, n_node, o_size, n_head)  # b_sz x n_node x n_head x o_size

        output = output.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, o_size)  # (b_sz*n_head) x n_node x o_size
        output = normalized_adj_t.bmm(output).view(bs, n_head, n_node, o_size).sum(1)  # b_sz x n_node x dv
        output = self.activation(output)
        output = self.dropout(output)
        return output


class RGCN(nn.Module):

    def __init__(self, input_size, num_heads, num_basis, num_layers, dropout, diag_decompose):
        super().__init__()
        self.layers = nn.ModuleList([RGCNLayer(num_heads, num_basis, input_size, input_size,
                                               dropout, diag_decompose=diag_decompose) for l in range(num_layers + 1)])

    def forward(self, inputs, adj):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        adj: tensor of shape (b_sz, n_head, n_node, n_node)
            we assume the identity matrix representating self loops are already added to adj
        """
        bs, n_head, n_node, _ = adj.size()

        in_degree = torch.max(adj.sum(2), adj.new_ones(()))
        adj_t = adj.transpose(2, 3)
        normalized_adj_t = (adj_t / in_degree.unsqueeze(3)).view(bs * n_head, n_node, n_node)
        assert ((torch.abs(normalized_adj_t.sum(2) - 1) < 1e-5) | (torch.abs(normalized_adj_t.sum(2)) < 1e-5)).all()

        output = inputs
        for layer in self.layers:
            output = layer(output, normalized_adj_t)
        return output


class RGCNNet(nn.Module):

    def __init__(self, num_concepts, num_relations, num_basis, sent_dim, concept_dim, concept_in_dim, freeze_ent_emb,
                 num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                 pretrained_concept_emb=None, diag_decompose=False, ablation=None):
        super().__init__()
        self.ablation = ablation

        self.concept_emb = CustomizedEmbedding(concept_num=num_concepts, concept_out_dim=concept_dim, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, use_contextualized=False)
        gnn_dim = concept_dim
        self.rgcn = RGCN(gnn_dim, num_relations, num_basis, num_gnn_layers, p_gnn, diag_decompose)
        self.pool_layer = MultiheadAttPoolLayer(num_attention_heads, sent_dim, gnn_dim)
        self.fc = MLP(gnn_dim + sent_dim, fc_dim, 1, num_fc_layers, p_fc, True)

    def forward(self, sent_vecs, concepts, adj, adj_lengths):
        """
        sent_vecs: (batch_size, d_sent)
        concepts: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)

        returns: (batch_size, 1)
        """
        bs, n_node = concepts.size()
        gnn_input = self.concept_emb(concepts)
        # node_type_embed = sent_vecs.new_zeros((bs, n_node, self.node_type_emb_dim))
        # gnn_input = torch.cat((gnn_input, node_type_embed), -1)
        gnn_output = self.rgcn(gnn_input, adj)

        adj_lengths = torch.max(adj_lengths, adj_lengths.new_ones(()))  # a temporary solution to avoid zero node
        mask = torch.arange(concepts.size(1), device=adj.device).unsqueeze(0) >= adj_lengths.unsqueeze(1)
        pooled, pool_attn = self.pool_layer(sent_vecs, gnn_output, mask)
        # pooled = sent_vecs.new_zeros((sent_vecs.size(0), self.hid2out.weight.size(1) - sent_vecs.size(1)))
        logits = self.fc(torch.cat((pooled, sent_vecs), 1))
        return logits, pool_attn


class LMRGCN(nn.Module):
    def __init__(self, model_name, num_concepts, num_relations, num_basis, concept_dim, concept_in_dim, freeze_ent_emb,
                 num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                 pretrained_concept_emb=None, diag_decompose=False, ablation=None, encoder_config={}):
        super().__init__()
        self.ablation = ablation
        self.model_name = model_name
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = RGCNNet(num_concepts, num_relations, num_basis, self.encoder.sent_dim, concept_dim, concept_in_dim, freeze_ent_emb,
                               num_gnn_layers, num_attention_heads, fc_dim, num_fc_layers, p_gnn, p_fc,
                               pretrained_concept_emb=pretrained_concept_emb, diag_decompose=diag_decompose, ablation=ablation)

    def forward(self, *inputs, layer_id=-1):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)

        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        *lm_inputs, concept_ids, node_type_ids, adj_lengths, adj = inputs
        if 'no_lm' not in self.ablation:
            sent_vecs, _ = self.encoder(*lm_inputs, layer_id=layer_id)
        else:
            sent_vecs = torch.ones((bs * nc, self.encoder.sent_dim), dtype=torch.float).to(concept_ids.device)
        logits, attn = self.decoder(sent_vecs=sent_vecs, concepts=concept_ids, adj=adj, adj_lengths=adj_lengths)
        logits = logits.view(bs, nc)
        return logits, attn


class LMRGCNDataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)

        self.num_choice = self.train_data[0].size(1)

        *train_extra_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, self.num_choice)
        self.train_data += train_extra_data
        *dev_extra_data, self.dev_adj_data, n_rel = load_adj_data(dev_adj_path, max_node_num, self.num_choice)
        self.dev_data += dev_extra_data
        assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data) == x.size(0) for x in [self.dev_labels] + self.dev_data)

        # pre-allocate an empty batch adj matrix
        self.adj_empty = torch.zeros((self.batch_size, self.num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32, device=device)
        self.eval_adj_empty = torch.zeros((self.eval_batch_size, self.num_choice, n_rel, max_node_num, max_node_num), dtype=torch.float32, device=device)

        if test_statement_path is not None:
            *test_extra_data, self.test_adj_data, n_rel = load_adj_data(test_adj_path, max_node_num, self.num_choice)
            self.test_data += test_extra_data
            assert all(len(self.test_qids) == len(self.test_adj_data) == x.size(0) for x in [self.test_labels] + self.test_data)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
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
        return AdjDataBatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                                     tensors=self.train_data, adj_empty=self.adj_empty, adj_data=self.train_adj_data)

    def train_eval(self):
        return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels,
                                     tensors=self.train_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)

    def dev(self):
        return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                     tensors=self.dev_data, adj_empty=self.eval_adj_empty, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return AdjDataBatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                         tensors=self.train_data, adj_empty=self.eval_adj_empty, adj_data=self.train_adj_data)
        else:
            return AdjDataBatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
                                         tensors=self.test_data, adj_empty=self.eval_adj_empty, adj_data=self.test_adj_data)
