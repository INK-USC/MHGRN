import dgl.function as fn
import networkx as nx
import torch.utils.data as data
from torch.nn import init

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
from utils.parser_utils import *

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GraphConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation):
        super(GraphConvLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class GCNEncoder(nn.Module):

    def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, concept_emd=None):
        super(GCNEncoder, self).__init__()

        self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
        self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)

        if pretrained_concept_emd is not None and concept_emd is None:
            self.concept_emd = nn.Embedding(pretrained_concept_emd.size(0), pretrained_concept_emd.size(1))
            self.concept_emd.weight.data.copy_(pretrained_concept_emd)
        elif pretrained_concept_emd is None and concept_emd is not None:
            self.concept_emd = concept_emd
        else:
            raise ValueError('invalid pretrained_concept_emd/concept_emd')

    def forward(self, g):
        features = self.concept_emd(g.ndata["cncpt_ids"])
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        g.ndata['h'] = x
        return g


class GCNLayer(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        assert input_size == output_size

        self.w = nn.Parameter(torch.zeros(input_size, output_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.normal_(self.w_vs, mean=0, std=np.sqrt(2.0 / (input_size + output_size)))

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, normalized_adj_t):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        normalized_adj_t: tensor of shape (b_sz, n_node, n_node)
            normalized_adj_t[:, j, i] ==  1/n indicates a directed edge i --> j and in_degree(j) == n
        """

        bs, n_node, _ = inputs.size()

        output = inputs.matmul(self.w)  # (b_sz, n_node, o_size)
        output = normalized_adj_t.bmm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class GCN(nn.Module):

    def __init__(self, input_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(input_size, input_size, dropout) for l in range(num_layers + 1)])

    def forward(self, inputs, adj):
        """
        inputs: tensor of shape (b_sz, n_node, d)
        adj: tensor of shape (b_sz, n_head, n_node, n_node)
            we assume the identity matrix representing self loops are already added to adj
        """
        bs, n_node, _ = adj.size()

        in_degree = torch.max(adj.sum(1), adj.new_ones(()))
        adj_t = adj.transpose(1, 2)
        normalized_adj_t = (adj_t / in_degree.unsqueeze(-1))  # (bz, n_node, n_node)
        assert ((torch.abs(normalized_adj_t.sum(2) - 1) < 1e-5) | (torch.abs(normalized_adj_t.sum(2)) < 1e-5)).all()

        output = inputs
        for layer in self.layers:
            output = layer(output, normalized_adj_t)
        return output


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class KnowledgeAwareGraphNetwork(nn.Module):

    def __init__(self, sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emd, pretrained_relation_emd,
                 lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True,
                 qa_attention=True):
        super(KnowledgeAwareGraphNetwork, self).__init__()
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.sent_dim = sent_dim
        self.concept_num = concept_num
        self.relation_num = relation_num
        self.qas_encoded_dim = qas_encoded_dim
        self.pretrained_concept_emd = pretrained_concept_emd
        self.pretrained_relation_emd = pretrained_relation_emd
        self.lstm_dim = lstm_dim
        self.lstm_layer_num = lstm_layer_num
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.dropout = dropout
        self.bidirect = bidirect
        self.num_random_paths = num_random_paths
        self.path_attention = path_attention
        self.qa_attention = qa_attention

        self.concept_emd = nn.Embedding(concept_num, concept_dim)
        self.relation_emd = nn.Embedding(relation_num, relation_dim)

        if pretrained_concept_emd is not None:
            self.concept_emd.weight.data.copy_(pretrained_concept_emd)
        else:
            bias = np.sqrt(6.0 / self.concept_dim)
            nn.init.uniform_(self.concept_emd.weight, -bias, bias)

        if pretrained_relation_emd is not None:
            self.relation_emd.weight.data.copy_(pretrained_relation_emd)
        else:
            bias = np.sqrt(6.0 / self.relation_dim)
            nn.init.uniform_(self.relation_emd.weight, -bias, bias)

        self.lstm = nn.LSTM(input_size=graph_output_dim + concept_dim + relation_dim,
                            hidden_size=lstm_dim,
                            num_layers=lstm_layer_num,
                            bidirectional=bidirect,
                            dropout=dropout,
                            batch_first=True)

        if bidirect:
            self.lstm_dim = lstm_dim * 2

        self.qas_encoder = nn.Sequential(
            nn.Linear(2 * (concept_dim + graph_output_dim) + sent_dim, self.qas_encoded_dim * 2),  # binary classification
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.qas_encoded_dim * 2, self.qas_encoded_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )

        if self.path_attention:  # TODO: can be optimized by using nn.BiLinaer
            self.qas_pathlstm_att = nn.Linear(self.qas_encoded_dim, self.lstm_dim)  # transform qas vector to query vectors
            self.qas_pathlstm_att.apply(weight_init)

        if self.qa_attention:
            self.sent_ltrel_att = nn.Linear(sent_dim, self.qas_encoded_dim)  # transform sentence vector to query vectors
            self.sent_ltrel_att.apply(weight_init)

        self.hidden2output = nn.Sequential(
            nn.Linear(self.qas_encoded_dim + self.lstm_dim + self.sent_dim, 1),  # binary classification
        )

        self.lstm.apply(weight_init)
        self.qas_encoder.apply(weight_init)
        self.hidden2output.apply(weight_init)

        self.graph_encoder = GCNEncoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
                                        pretrained_concept_emd=None, concept_emd=self.concept_emd)

    def forward(self, s_vec_batched, qa_pairs_batched, cpt_paths_batched, rel_paths_batched, qa_path_num_batched, path_len_batched, graphs, concept_mapping_dicts, ana_mode=False):
        output_graphs = self.graph_encoder(graphs)
        new_concept_embed = torch.cat((output_graphs.ndata["h"], s_vec_batched.new_zeros((1, self.graph_output_dim))))  # len(output_concept_embeds) as padding

        final_vecs = []

        if ana_mode:
            path_att_scores = []
            qa_pair_att_scores = []

        n_qa_pairs = [len(t) for t in qa_pairs_batched]
        total_qa_pairs = sum(n_qa_pairs)
        s_vec_expanded = s_vec_batched.new_zeros((total_qa_pairs, s_vec_batched.size(1)))
        i = 0
        for n, s_vec in zip(n_qa_pairs, s_vec_batched):
            j = i + n
            s_vec_expanded[i:j] = s_vec
            i = j
        qa_ids_batched = torch.cat(qa_pairs_batched, 0)  # N x 2
        qa_vecs = self.concept_emd(qa_ids_batched).view(total_qa_pairs, -1)
        new_qa_ids = []
        for qa_ids, mdict in zip(qa_pairs_batched, concept_mapping_dicts):
            id_mapping = lambda x: mdict.get(x, len(new_concept_embed) - 1)
            new_qa_ids += [[id_mapping(q), id_mapping(a)] for q, a in qa_ids]
        new_qa_ids = torch.tensor(new_qa_ids, device=s_vec_batched.device)
        new_qa_vecs = new_concept_embed[new_qa_ids].view(total_qa_pairs, -1)
        raw_qas_vecs = torch.cat((qa_vecs, new_qa_vecs, s_vec_expanded), dim=1)  # all the qas triple vectors associated with a statement
        qas_vecs_batched = self.qas_encoder(raw_qas_vecs)
        if self.path_attention:
            query_vecs_batched = self.qas_pathlstm_att(qas_vecs_batched)
        flat_cpt_paths_batched = torch.cat(cpt_paths_batched, 0)
        mdicted_cpaths = []
        for cpt_path in flat_cpt_paths_batched:
            mdicted_cpaths.append([id_mapping(c) for c in cpt_path])
        mdicted_cpaths = torch.tensor(mdicted_cpaths, device=s_vec_batched.device)

        new_batched_all_qa_cpt_paths_embeds = new_concept_embed[mdicted_cpaths]
        batched_all_qa_cpt_paths_embeds = self.concept_emd(torch.cat(cpt_paths_batched, 0))  # old concept embed

        batched_all_qa_cpt_paths_embeds = torch.cat((batched_all_qa_cpt_paths_embeds, new_batched_all_qa_cpt_paths_embeds), 2)

        batched_all_qa_rel_paths_embeds = self.relation_emd(torch.cat(rel_paths_batched, 0))  # N_PATHS x D x MAX_PATH_LEN

        batched_all_qa_cpt_rel_path_embeds = torch.cat((batched_all_qa_cpt_paths_embeds,
                                                        batched_all_qa_rel_paths_embeds), 2)

        # if False then abiliate the LSTM
        if True:
            batched_lstm_outs, _ = self.lstm(batched_all_qa_cpt_rel_path_embeds)

        else:
            batched_lstm_outs = s_vec.new_zeros((batched_all_qa_cpt_rel_path_embeds.size(0),
                                                 batched_all_qa_cpt_rel_path_embeds.size(1),
                                                 self.lstm_dim))
        b_idx = torch.arange(batched_lstm_outs.size(0)).to(batched_lstm_outs.device)
        batched_lstm_outs = batched_lstm_outs[b_idx, torch.cat(path_len_batched, 0) - 1, :]

        qa_pair_cur_start = 0
        path_cur_start = 0
        # for each question-answer statement
        for s_vec, qa_ids, cpt_paths, rel_paths, mdict, qa_path_num, path_len in zip(s_vec_batched, qa_pairs_batched, cpt_paths_batched,
                                                                                     rel_paths_batched, concept_mapping_dicts, qa_path_num_batched,
                                                                                     path_len_batched):  # len = batch_size * num_choices

            n_qa_pairs = qa_ids.size(0)
            qa_pair_cur_end = qa_pair_cur_start + n_qa_pairs

            if n_qa_pairs == 0 or False:  # if "or True" then we can do ablation study
                raw_qas_vecs = torch.cat([s_vec.new_zeros((self.concept_dim + self.graph_output_dim) * 2), s_vec], 0).view(1, -1)
                qas_vecs = self.qas_encoder(raw_qas_vecs)
                latent_rel_vecs = torch.cat((qas_vecs, s_vec.new_zeros(1, self.lstm_dim)), dim=1)
            else:
                pooled_path_vecs = []
                qas_vecs = qas_vecs_batched[qa_pair_cur_start:qa_pair_cur_end]
                for j in range(n_qa_pairs):
                    if self.path_attention:
                        query_vec = query_vecs_batched[qa_pair_cur_start + j]

                    path_cur_end = path_cur_start + qa_path_num[j]

                    # pooling over all paths for a certain (question concept, answer concept) pair
                    blo = batched_lstm_outs[path_cur_start:path_cur_end]
                    if self.path_attention:  # TODO: use an attention module for better readability
                        att_scores = torch.mv(blo, query_vec)  # path-level attention scores
                        norm_att_scores = F.softmax(att_scores, 0)
                        att_pooled_path_vec = torch.mv(blo.t(), norm_att_scores)
                        if ana_mode:
                            path_att_scores.append(norm_att_scores)
                    else:
                        att_pooled_path_vec = blo.mean(0)

                    path_cur_start = path_cur_end
                    pooled_path_vecs.append(att_pooled_path_vec)

                pooled_path_vecs = torch.stack(pooled_path_vecs, 0)
                latent_rel_vecs = torch.cat((qas_vecs, pooled_path_vecs), 1)  # qas and KE-qas

            # pooling over all (question concept, answer concept) pairs
            if self.path_attention:
                sent_as_query = self.sent_ltrel_att(s_vec)  # sent attend on qas
                r_att_scores = torch.mv(qas_vecs, sent_as_query)  # qa-pair-level attention scores
                norm_r_att_scores = F.softmax(r_att_scores, 0)
                if ana_mode:
                    qa_pair_att_scores.append(norm_r_att_scores)
                final_vec = torch.mv(latent_rel_vecs.t(), norm_r_att_scores)
            else:
                final_vec = latent_rel_vecs.mean(0).to(s_vec.device)  # mean pooling
            final_vecs.append(torch.cat((final_vec, s_vec), 0))

            qa_pair_cur_start = qa_pair_cur_end

        logits = self.hidden2output(torch.stack(final_vecs))
        if not ana_mode:
            return logits
        else:
            return logits, (path_att_scores, qa_pair_att_scores)


class LMKagNet(nn.Module):
    # qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts
    def __init__(self, model_name, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emb, pretrained_relation_emb,
                 lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True,
                 qa_attention=True, encoder_config={}):
        super().__init__()

        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = KnowledgeAwareGraphNetwork(self.encoder.sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                                                  qas_encoded_dim, pretrained_concept_emb, pretrained_relation_emb,
                                                  lstm_dim, lstm_layer_num, graph_hidden_dim, graph_output_dim,
                                                  dropout=dropout, bidirect=bidirect, num_random_paths=num_random_paths, path_attention=path_attention,
                                                  qa_attention=qa_attention)

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
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-7]] + inputs[-7:]  # merge the batch dimension and the num_choice dimension
        print(len(inputs))
        *lm_inputs, qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts = inputs
        print([len(x[0]) for x in [qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts]])
        print([x.device for x in [qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts]])
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.view(bs, nc, -1).to(qa_pair_data.device), qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data, batched_graph, concept_mapping_dicts)
        logits = logits.view(bs, nc)
        return logits, attn


class KagNetDataLoader(data.Dataset):

    def __init__(self, train_statement_path: str, train_path_jsonl: str, train_ngx_jsonl: str, dev_statement_path: str,
                 dev_path_jsonl: str, dev_ngx_jsonl: str, test_statement_path: str, test_path_jsonl: str, test_ngx_jsonl: str,
                 batch_size, eval_batch_size, device, max_path_len=5, max_seq_length=128, model_name=None,
                 is_inhouse=True, inhouse_train_qids_path=None, use_cache=True, format=[]):
        super(KagNetDataLoader, self).__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.vocab = None
        model_type = MODEL_NAME_TO_CLASS.get(model_name, 'lstm')

        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
        self.num_choice = self.train_encoder_data[0].size(1)

        # self.qa_pair_data, self.cpt_path_data, self.rel_path_data, self.qa_path_num_data, self.path_len_data
        self.train_decoder_list_data = list(self._load_paths_data(train_path_jsonl, max_path_len, use_cache))
        self.dev_decoder_list_data = list(self._load_paths_data(dev_path_jsonl, max_path_len, use_cache))
        if test_statement_path is not None:
            self.test_decoder_list_data = list(self._load_paths_data(test_path_jsonl, max_path_len, use_cache))

        # self.nxgs
        self.train_graph_data = self._load_graphs(train_ngx_jsonl, use_cache)
        self.dev_graph_data = self._load_graphs(dev_ngx_jsonl, use_cache)
        if test_statement_path is not None:
            self.test_graph_data = self._load_graphs(test_ngx_jsonl, use_cache)
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_encoder_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data)
        if test_statement_path is not None:
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_encoder_data)

    def _load_paths_data(self, pf_jsonl, max_path_len, use_cache):  # load all qa and paths
        # use_cache = False
        save_file = pf_jsonl + ".pk"
        if use_cache and os.path.exists(save_file):
            print(f'using cached paths from {save_file}')
            with open(save_file, 'rb') as handle:
                qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data = pickle.load(handle)
            return qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data

        nrow = sum(1 for _ in open(pf_jsonl, 'r'))
        qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data = [], [], [], [], []
        with open(pf_jsonl, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, total=nrow, desc="loading paths"):
                s = json.loads(line)
                qa_pairs, paths, rels, qa_path_num, path_len = [], [], [], [], []
                for qas in s:  # iterate over all (question concept, answer concept) pairs
                    pf_res = qas["pf_res"]
                    if pf_res is not None:
                        for item in pf_res:
                            p = item["path"]
                            r = item["rel"]
                            q = p[0]
                            a = p[-1]
                            new_qa_pair = (q, a) not in qa_pairs
                            if new_qa_pair:
                                qa_pairs.append((q, a))
                                qa_path_num.append(0)

                            if len(p) > max_path_len and not new_qa_pair:
                                continue  # cut off by length of concepts
                            assert len(p) - 1 == len(r)
                            path_len.append(len(p))

                            p += [0] * (max_path_len - len(p))  # padding
                            for i in range(len(r)):
                                for j in range(len(r[i])):
                                    if r[i][j] - 17 in r[i]:
                                        r[i][j] -= 17  # to delete realtedto* and antonym*

                            r = [n[0] for n in r]  # only pick the top relation while multiple ones are okay
                            r += [0] * (max_path_len - len(r))  # padding
                            paths.append(p)
                            rels.append(r)
                            qa_path_num[-1] += 1

                qa_pair_data.append(torch.tensor(qa_pairs) if qa_pairs else torch.zeros((0, 2), dtype=torch.int64))
                cpt_path_data.append(torch.tensor(paths) if paths else torch.zeros((0, max_path_len), dtype=torch.int64))
                rel_path_data.append(torch.tensor(rels) if rels else torch.zeros((0, max_path_len), dtype=torch.int64))
                qa_path_num_data.append(torch.tensor(qa_path_num) if qa_path_num else torch.zeros(0, dtype=torch.int64))
                path_len_data.append(torch.tensor(path_len) if path_len else torch.zeros(0, dtype=torch.int64))

        qa_pair_data = list(map(list, zip(*(iter(qa_pair_data),) * self.num_choice)))
        cpt_path_data = list(map(list, zip(*(iter(cpt_path_data),) * self.num_choice)))
        rel_path_data = list(map(list, zip(*(iter(rel_path_data),) * self.num_choice)))
        qa_path_num_data = list(map(list, zip(*(iter(qa_path_num_data),) * self.num_choice)))
        path_len_data = list(map(list, zip(*(iter(path_len_data),) * self.num_choice)))

        print([x[0].size() for x in [qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data]])

        with open(save_file, 'wb') as fout:
            pickle.dump([qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data], fout)

        return qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data

    def _load_graphs(self, graph_ngx_jsonl, use_cache):
        save_file = graph_ngx_jsonl + ".pk"
        if use_cache and os.path.exists(save_file):
            print(f'using cached graphs from {save_file}')
            with open(save_file, 'rb') as fin:
                dgs = pickle.load(fin)
            return dgs

        dgs = []
        with open(graph_ngx_jsonl, 'r', encoding='utf-8') as fin:
            nxgs = [line for line in fin]
        for nxg_str in tqdm(nxgs, total=len(nxgs), desc='loading graphs'):
            nxg = nx.node_link_graph(json.loads(nxg_str))
            dg = dgl.DGLGraph(multigraph=True)
            dg.from_networkx(nxg)
            cids = [nxg.nodes[n_id]['cid'] for n_id in range(len(dg))]
            dg.ndata.update({'cncpt_ids': torch.tensor(cids)})
            dgs.append(dg)

        dgs = list(map(list, zip(*(iter(dgs),) * self.num_choice)))

        with open(save_file, 'wb') as fout:
            pickle.dump(dgs, fout)

        return dgs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        raise NotImplementedError

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
        return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, lists1=self.train_decoder_list_data, graph_data=self.train_graph_data)

    def train_eval(self):
        return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels,
                                             tensors0=self.train_encoder_data, lists1=self.train_decoder_list_data, graph_data=self.train_graph_data)

    def dev(self):
        return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels,
                                             tensors0=self.dev_encoder_data, lists1=self.dev_decoder_list_data, graph_data=self.dev_graph_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels,
                                                 tensors0=self.train_encoder_data, lists1=self.train_decoder_list_data, graph_data=self.train_graph_data)
        else:
            return MultiGPUNxgDataBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels,
                                                 tensors0=self.test_encoder_data, lists1=self.test_decoder_list_data, graph_data=self.test_graph_data)
