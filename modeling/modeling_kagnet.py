import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import init
import dgl
import dgl.function as fn
import numpy as np
import json
from tqdm import tqdm
import pickle
import os
import networkx as nx

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


class GCNSent(nn.Module):

    def __init__(self, sent_dim, fc_hidden_size, concept_dim, graph_hidden_dim, graph_output_dim,
                 pretrained_concept_emd, dropout=0.3):
        super(GCNSent, self).__init__()
        self.sent_dim = sent_dim
        self.fc_hidden_size = fc_hidden_size
        self.concept_dim = concept_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.pretrained_concept_emd = pretrained_concept_emd
        self.dropout = dropout

        self.graph_encoder = GCNEncoder(concept_dim, graph_hidden_dim, graph_output_dim, pretrained_concept_emd)

        self.mlp = nn.Sequential(
            nn.Linear(self.sent_dim + graph_output_dim, self.fc_hidden_size * 4),
            nn.BatchNorm1d(self.fc_hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden_size * 4, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fc_hidden_size, 1),
        )

    def forward(self, sent_vecs, graph):
        node_embed = self.graph_encoder(graph)
        graph_embed = dgl.mean_nodes(node_embed, 'h')
        concated = torch.cat((sent_vecs, graph_embed), 1)
        logits = self.mlp(concated)
        return logits


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


class RelationNetwork(nn.Module):

    def __init__(self, concept_dim, concept_num, pretrained_concept_emd, sent_dim, latent_rel_dim, device):

        super(RelationNetwork, self).__init__()
        self.concept_dim = concept_dim
        self.concept_num = concept_num
        self.pretrained_concept_emd = pretrained_concept_emd
        self.sent_dim = sent_dim
        self.latent_rel_dim = latent_rel_dim
        self.device = device

        self.concept_emd = nn.Embedding(concept_num, concept_dim)
        if pretrained_concept_emd is not None:
            self.concept_emd.weight.data.copy_(pretrained_concept_emd)
        else:
            bias = np.sqrt(6.0 / self.concept_dim)
            nn.init.uniform_(self.concept_emd.weight, -bias, bias)

        self.relation_extractor = nn.Sequential(
            nn.Linear(concept_dim * 2 + sent_dim, self.latent_rel_dim * 2),  # binary classification
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_rel_dim * 2),
            nn.Linear(self.latent_rel_dim * 2, self.latent_rel_dim),
            nn.BatchNorm1d(self.latent_rel_dim),
            nn.ReLU(),
        )

        self.hidden2output = nn.Sequential(
            nn.Linear(latent_rel_dim, 1),  # binary classification
        )

    def forward(self, statement_vecs, qa_pairs):
        # statement_vecs = statement_vecs.to(self.device)
        qa_pooled_vecs = []
        for qa_ids in qa_pairs:
            qa_ids = qa_ids.to(self.device)
            if qa_ids.size(0) == 0 or False:  # if True then abaliate qa pairs
                qa_pooled_vecs.append(statement_vecs.new_zeros((self.concept_dim * 2,)))
            else:
                qa_embed = self.concept_emd(qa_ids)  # N x 2 x D
                qa_pooled = qa_embed.mean(0).view(-1)  # 2D
                qa_pooled_vecs.append(qa_pooled)
        qa_pooled_vecs = torch.stack(qa_pooled_vecs, 0)
        # each qas_vec is the concat of the question concept, answer concept, and the statement
        qas_vecs = torch.cat([qa_pooled_vecs, statement_vecs], 1)
        latent_rel_vecs = self.relation_extractor(qas_vecs)
        logits = self.hidden2output(latent_rel_vecs)
        return logits


class KnowledgeEnhancedRelationNetwork(nn.Module):

    def __init__(self, sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emd, pretrained_relation_emd,
                 lstm_dim, lstm_layer_num, device,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True, qa_attention=True):

        super(KnowledgeEnhancedRelationNetwork, self).__init__()
        self.sent_dim = sent_dim
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.concept_num = concept_num
        self.relation_num = relation_num
        self.qas_encoded_dim = qas_encoded_dim
        self.pretrained_concept_emd = pretrained_concept_emd
        self.pretrained_relation_emd = pretrained_relation_emd
        self.lstm_dim = lstm_dim
        self.lstm_layer_num = lstm_layer_num
        self.device = device
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

        self.lstm = nn.LSTM(input_size=concept_dim + relation_dim,
                            hidden_size=lstm_dim,
                            num_layers=lstm_layer_num,
                            bidirectional=bidirect,
                            dropout=dropout,
                            batch_first=True)

        if bidirect:
            self.lstm_dim = lstm_dim * 2

        self.qas_encoder = nn.Sequential(
            nn.Linear(2 * concept_dim + sent_dim, self.qas_encoded_dim * 2),  # binary classification
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

    def forward(self, s_vec_batched, qa_pairs_batched, cpt_paths_batched, rel_paths_batched,
                qa_path_num_batched, path_len_batched, ana_mode=False):
        final_vecs = []

        if ana_mode:
            path_att_scores = []
            qa_pair_att_scores = []

        # for each question-answer statement
        for s_vec, qa_ids, cpt_paths, rel_paths, qa_path_num, path_len in zip(s_vec_batched, qa_pairs_batched, cpt_paths_batched,
                                                                              rel_paths_batched, qa_path_num_batched, path_len_batched):  # len = batch_size * num_choices
            n_qa_pairs = qa_ids.size(0)
            if n_qa_pairs == 0 or False:  # if "or True" then we can do abalation study
                raw_qas_vecs = torch.cat([s_vec.new_zeros(self.concept_dim * 2), s_vec], 0).view(1, -1)
                qas_vecs = self.qas_encoder(raw_qas_vecs)
                latent_rel_vecs = torch.cat((qas_vecs, s_vec.new_zeros(1, self.lstm_dim)), dim=1)
            else:
                qa_vecs = self.concept_emd(qa_ids).view(n_qa_pairs, -1)
                s_vecs = s_vec.view(1, -1).expand(n_qa_pairs, -1)
                raw_qas_vecs = torch.cat((qa_vecs, s_vecs), dim=1)  # all the qas triple vectors associated with a statement
                qas_vecs = self.qas_encoder(raw_qas_vecs)

                batched_all_qa_cpt_paths_embeds = self.concept_emd(cpt_paths)  # N_PATHS x MAX_PATH_LEN x D
                batched_all_qa_rel_paths_embeds = self.relation_emd(rel_paths)  # N_PATHS x MAX_PATH_LEN x D
                batched_all_qa_cpt_rel_path_embeds = torch.cat((batched_all_qa_cpt_paths_embeds,
                                                                batched_all_qa_rel_paths_embeds), dim=2)

                # if False then abiliate the LSTM
                if True:
                    batched_lstm_outs, _ = self.lstm(batched_all_qa_cpt_rel_path_embeds)
                else:
                    batched_lstm_outs = s_vec.new_zeros((batched_all_qa_cpt_rel_path_embeds.size(0),
                                                         batched_all_qa_cpt_rel_path_embeds.size(1),
                                                         self.lstm_dim))
                b_idx = torch.arange(batched_lstm_outs.size(0)).to(batched_lstm_outs.device)
                batched_lstm_outs = batched_lstm_outs[b_idx, path_len - 1, :]

                if self.path_attention:
                    query_vecs = self.qas_pathlstm_att(qas_vecs)

                pooled_path_vecs = []
                cur_start = 0
                for j in range(n_qa_pairs):
                    if self.path_attention:
                        query_vec = query_vecs[j]
                    cur_end = cur_start + qa_path_num[j]

                    # pooling over all paths for a certain (question concept, answer concept) pair
                    blo = batched_lstm_outs[cur_start:cur_end]
                    if self.path_attention:  # TODO: use an attention module for better readibility
                        att_scores = torch.mv(blo, query_vec)  # path-level attention scores
                        norm_att_scores = F.softmax(att_scores, 0)
                        att_pooled_path_vec = torch.mv(blo.t(), norm_att_scores)
                        if ana_mode:
                            path_att_scores.append(norm_att_scores)
                    else:
                        att_pooled_path_vec = blo.mean(0)

                    cur_start = cur_end
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
                final_vec = latent_rel_vecs.mean(0).to(self.device)  # mean pooling
            final_vecs.append(torch.cat((final_vec, s_vec), 0))

        logits = self.hidden2output(torch.stack(final_vecs))
        if not ana_mode:
            return logits
        else:
            return logits, path_att_scores, qa_pair_att_scores


class KnowledgeAwareGraphNetwork(nn.Module):

    def __init__(self, sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                 qas_encoded_dim, pretrained_concept_emd, pretrained_relation_emd,
                 lstm_dim, lstm_layer_num, device, graph_hidden_dim, graph_output_dim,
                 dropout=0.1, bidirect=True, num_random_paths=None, path_attention=True,
                 qa_attention=True):
        super(KnowledgeAwareGraphNetwork, self).__init__()
        self.sent_dim = sent_dim
        self.concept_dim = concept_dim
        self.relation_dim = relation_dim
        self.concept_num = concept_num
        self.relation_num = relation_num
        self.qas_encoded_dim = qas_encoded_dim
        self.pretrained_concept_emd = pretrained_concept_emd
        self.pretrained_relation_emd = pretrained_relation_emd
        self.lstm_dim = lstm_dim
        self.lstm_layer_num = lstm_layer_num
        self.device = device
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

    def forward(self, s_vec_batched, qa_pairs_batched, cpt_paths_batched, rel_paths_batched,
                graphs, concept_mapping_dicts, qa_path_num_batched, path_len_batched, ana_mode=False):
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
        new_qa_ids = torch.tensor(new_qa_ids).to(self.device)
        new_qa_vecs = new_concept_embed[new_qa_ids].view(total_qa_pairs, -1)
        raw_qas_vecs = torch.cat((qa_vecs, new_qa_vecs, s_vec_expanded), dim=1)  # all the qas triple vectors associated with a statement
        qas_vecs_batched = self.qas_encoder(raw_qas_vecs)
        if self.path_attention:
            query_vecs_batched = self.qas_pathlstm_att(qas_vecs_batched)

        flat_cpt_paths_batched = torch.cat(cpt_paths_batched, 0)
        mdicted_cpaths = []
        for cpt_path in flat_cpt_paths_batched:
            mdicted_cpaths.append([id_mapping(c) for c in cpt_path])
        mdicted_cpaths = torch.tensor(mdicted_cpaths).to(self.device)

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
                    if self.path_attention:  # TODO: use an attention module for better readibility
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
                final_vec = latent_rel_vecs.mean(0).to(self.device)  # mean pooling
            final_vecs.append(torch.cat((final_vec, s_vec), 0))

            qa_pair_cur_start = qa_pair_cur_end

        logits = self.hidden2output(torch.stack(final_vecs))
        if not ana_mode:
            return logits
        else:
            return logits, path_att_scores, qa_pair_att_scores


class KagNetMLP(nn.Module):

    def __init__(self, sent_dim=1024, hidden_dim=64, out_dim=1, dropout=0.):
        super().__init__()
        self.in_dim = sent_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(sent_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, sent):
        return self.fc(sent)


class CSQADataLoader(data.Dataset):

    def __init__(self, statement_jsonl: str, pf_jsonl: str, graph_ngx_jsonl: str, pretrained_sent_vecs: str,
                 batch_size, device, shuffle=True, num_choice=5, max_path_len=5, start=0, end=None, cut_off=3,
                 is_test=False, use_cache=True):
        super(CSQADataLoader, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.is_test = is_test

        self.qids, self.statement_vecs, self.correct_labels, \
        self.qa_text = self._load_statement_data(statement_jsonl, pretrained_sent_vecs, num_choice)
        self.qa_pair_data, self.cpt_path_data, self.rel_path_data, self.qa_path_num_data, self.path_len_data = \
            self._load_paths_data(pf_jsonl, num_choice, cut_off, max_path_len, use_cache)
        self.nxgs, self.dgs = self._load_graphs(graph_ngx_jsonl, num_choice, use_cache)

        for name in ['qids', 'statement_vecs', 'correct_labels', 'qa_text', 'cpt_path_data',
                     'rel_path_data', 'qa_pair_data', 'nxgs', 'dgs']:
            obj = getattr(self, name)
            setattr(self, name, obj[start:end])

        assert len(self.qids) == len(self.statement_vecs) \
               == len(self.correct_labels) == len(self.qa_text) \
               == len(self.cpt_path_data) == len(self.rel_path_data) \
               == len(self.qa_pair_data) == len(self.qa_path_num_data) \
               == len(self.path_len_data) == len(self.nxgs) == len(self.dgs)
        self.n_samples = len(self.qids)

        if shuffle and not is_test:
            self.permutation = torch.randperm(self.n_samples)
        else:
            self.permutation = torch.arange(self.n_samples)

    def _load_statement_data(self, statement_jsonl, pretrained_sent_vecs, num_choice):
        with open(statement_jsonl, 'r') as fin:
            statement_data = [json.loads(line) for line in fin]

        n = len(statement_data)
        statement_vecs = np.load(pretrained_sent_vecs).reshape(n, num_choice, -1)
        statement_vecs = torch.tensor(statement_vecs)

        qids, correct_labels, qa_text = [], [], []
        for i in tqdm(range(n), total=n, desc='loading statements'):
            qa_text_cur = []
            qids.append(statement_data[i]["id"])
            for j, s in enumerate(statement_data[i]["statements"]):
                assert len(statement_data[i]["statements"]) == num_choice  # 5
                qa_text_cur.append((s["statement"], s['label']))
                if s["label"] is True:  # true of false
                    correct_labels.append(j)  # the truth id [0,1,2,3,4]
            qa_text.append(qa_text_cur)

        correct_labels = torch.tensor(correct_labels)
        return qids, statement_vecs, correct_labels, qa_text

    def _load_paths_data(self, pf_jsonl, num_choice, cut_off, max_path_len, use_cache):  # load all qa and paths
        save_file = pf_jsonl + ".pk"
        if use_cache and os.path.exists(save_file):
            print(f'using cached paths from {save_file}')
            with open(save_file, 'rb') as handle:
                qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data = pickle.load(handle)
            return qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data

        nrow = sum(1 for _ in open(pf_jsonl, 'r'))
        qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data = [], [], [], [], []
        with open(pf_jsonl, 'r') as fin:
            for line in tqdm(fin, total=nrow, desc="loading paths"):
                s = json.loads(line)
                qa_pairs, paths, rels, qa_path_num, path_len = [], [], [], [], []
                for qas in s:  # iterate over all (question concept, answer concept) pairs
                    pf_res = qas["pf_res"]
                    if pf_res is not None:
                        for item in pf_res:
                            p = item["path"]
                            r = item["rel"]
                            q = p[0] + 1
                            a = p[-1] + 1
                            new_qa_pair = (q, a) not in qa_pairs
                            if new_qa_pair:
                                qa_pairs.append((q, a))
                                qa_path_num.append(0)

                            if len(p) > cut_off and not new_qa_pair:
                                continue  # cut off by length of concepts

                            assert len(p) - 1 == len(r)
                            path_len.append(len(p))

                            p = [n + 1 for n in p]
                            p += [0] * (max_path_len - len(p))  # padding

                            for i in range(len(r)):
                                for j in range(len(r[i])):
                                    if r[i][j] - 17 in r[i]:
                                        r[i][j] -= 17  # to delete realtedto* and antonym*

                            r = [n[0] + 1 for n in r]  # only pick the top relation while multiple ones are okay
                            r += [0] * (max_path_len - len(r))  # padding
                            paths.append(p)
                            rels.append(r)
                            qa_path_num[-1] += 1

                qa_pair_data.append(torch.tensor(qa_pairs) if qa_pairs else torch.zeros((0, 2), dtype=torch.int64))
                cpt_path_data.append(torch.tensor(paths) if paths else torch.zeros((0, max_path_len), dtype=torch.int64))
                rel_path_data.append(torch.tensor(rels) if rels else torch.zeros((0, max_path_len), dtype=torch.int64))
                qa_path_num_data.append(qa_path_num if qa_path_num else torch.zeros(0, dtype=torch.int64))
                path_len_data.append(torch.tensor(path_len) if path_len else torch.zeros(0, dtype=torch.int64))

        qa_pair_data = list(map(list, zip(*(iter(qa_pair_data),) * num_choice)))
        cpt_path_data = list(map(list, zip(*(iter(cpt_path_data),) * num_choice)))
        rel_path_data = list(map(list, zip(*(iter(rel_path_data),) * num_choice)))
        qa_path_num_data = list(map(list, zip(*(iter(qa_path_num_data),) * num_choice)))
        path_len_data = list(map(list, zip(*(iter(path_len_data),) * num_choice)))

        with open(save_file, 'wb') as fout:
            pickle.dump([qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data], fout)

        return qa_pair_data, cpt_path_data, rel_path_data, qa_path_num_data, path_len_data

    def _load_graphs(self, graph_ngx_jsonl, num_choice, use_cache):
        save_file = graph_ngx_jsonl + ".pk"
        if use_cache and os.path.exists(save_file):
            print(f'using cached graphs from {save_file}')
            with open(save_file, 'rb') as fin:
                nxgs, dgs = pickle.load(fin)
            return nxgs, dgs

        dgs = []
        with open(graph_ngx_jsonl, 'r') as fin:
            nxgs = [line for line in fin]
        for nxg_str in tqdm(nxgs, total=len(nxgs), desc='loading graphs'):
            nxg = nx.node_link_graph(json.loads(nxg_str))
            dg = dgl.DGLGraph(multigraph=True)
            dg.from_networkx(nxg)
            cids = [nxg.nodes[n_id]['cid'] + 1 for n_id in range(len(dg))]  # -1 --> 0 and 0 stands for a placeholder concept
            dg.ndata.update({'cncpt_ids': torch.tensor(cids)})
            dgs.append(dg)

        nxgs = list(map(list, zip(*(iter(nxgs),) * num_choice)))
        dgs = list(map(list, zip(*(iter(dgs),) * num_choice)))

        with open(save_file, 'wb') as fout:
            pickle.dump([nxgs, dgs], fout)

        return nxgs, dgs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.statements[index], self.correct_labels[index], self.dgs[index], \
               self.cpt_path_data[index], self.rel_path_data[index], \
               self.qa_pair_data[index], self.qa_text[index]

    def __iter__(self):
        """
        generates a batch of data
        """

        def to_device(obj):
            if isinstance(obj, (tuple, list)):
                return [to_device(item) for item in obj]
            else:
                return obj.to(self.device)

        for i in range(0, self.n_samples, self.batch_size):
            j = min(self.n_samples, i + self.batch_size)
            indexes = self.permutation[i:j]

            sents_vecs = to_device(self.statement_vecs[indexes])
            labels = to_device(self.correct_labels[indexes])
            cpt_path_data = to_device([self.cpt_path_data[idx] for idx in indexes])
            rel_path_data = to_device([self.rel_path_data[idx] for idx in indexes])
            qa_pair_data = to_device([self.qa_pair_data[idx] for idx in indexes])
            graph_data = [self.dgs[idx] for idx in indexes]
            qa_path_num = [self.qa_path_num_data[idx] for idx in indexes]
            path_len_data = to_device([self.path_len_data[idx] for idx in indexes])

            # merge graphs into a single batched graph
            flat_graph_data = sum(graph_data, [])
            concept_mapping_dicts = []
            acc_start = 0
            for g in flat_graph_data:
                concept_mapping_dict = {}
                for index, cncpt_id in enumerate(g.ndata['cncpt_ids']):
                    concept_mapping_dict[int(cncpt_id)] = acc_start + index
                acc_start += len(g.nodes())
                concept_mapping_dicts.append(concept_mapping_dict)
            batched_graph = dgl.batch(flat_graph_data)
            batched_graph.ndata['cncpt_ids'] = batched_graph.ndata['cncpt_ids'].to(self.device)

            yield sents_vecs, labels, batched_graph, cpt_path_data, rel_path_data, \
                  qa_pair_data, concept_mapping_dicts, qa_path_num, path_len_data

    def reshuffle(self):
        self.permutation = torch.randperm(self.n_samples)
