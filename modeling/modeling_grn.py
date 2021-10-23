from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *


class MultiHopMessagePassingLayerDeprecated(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size, n_head))
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size, hidden_size * n_head))
        else:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size * hidden_size, n_basis))
            self.w_vs_co = nn.Parameter(torch.randn(k, n_basis, n_head))

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        ablation: list[str]
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        if self.diag_decompose:
            W = self.w_vs
        elif self.n_basis == 0:
            W = self.w_vs
        else:
            W = self.w_vs.bmm(self.w_vs_co).view(k, h_size, h_size * n_head)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)
        if 'no_trans' in self.ablation or 'no_att' in self.ablation:
            Z = X * start_attn.unsqueeze(2)
            for t in range(k):
                if self.diag_decompose:
                    Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_node, h_size, n_head)
                else:
                    Z = Z.matmul(W[t]).view(bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, None, None]
                Z = A.bmm(Z)
                Z = Z.view(bs, n_head, n_node, h_size).sum(1)
            Z = Z * end_attn.unsqueeze(2)

            D = start_attn.clone()
            for t in range(k):
                D = D.repeat(1, n_head).view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, None, None]
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
                    Z = Z * trans_attn[:, :, None, None, :]
                    Z = Z.sum(1)  # (bs, n_node, h_size,n_head)

                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, None, None]
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
                    D = D.unsqueeze(2) * trans_attn.unsqueeze(3)
                    D = D.sum(1)
                D = D.contiguous().view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, None, None]
                D = A.bmm(D)
                D = D.view(bs, n_head, n_node)
            if k >= 1:
                D = D.sum(1)
            D = D * end_attn  # (bs, n_node)

        Z = Z / (D.unsqueeze(2) + self.eps)
        return Z


class MultiHopMessagePassingLayer(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, init_range=0.01, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose and n_basis > 0:
            raise ValueError('diag_decompose and n_basis > 0 cannot be True at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size, n_head + 1))  # the additional head is used for the self-loop
            self.w_vs.data.uniform_(-init_range, init_range)
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(k, n_head + 1, hidden_size, hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
        else:
            self.w_vs = nn.Parameter(torch.zeros(k, n_basis, hidden_size * hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
            self.w_vs_co = nn.Parameter(torch.zeros(k, n_head + 1, n_basis))
            self.w_vs_co.data.uniform_(-init_range, init_range)

    def init_from_old(self, w_vs, w_vs_co=None):
        """
        w_vs: tensor of shape (k, h_size, n_head) or (k, h_size, h_size * n_head) or (k, h_size*h_size, n_basis)
        w_vs_co: tensor of shape (k, n_basis, n_head)
        """
        raise NotImplementedError()
        k, n_head, h_size = self.k, self.n_head, self.hidden_size
        if self.diag_decompose:
            self.w_vs.data.copy_(w_vs)
        elif self.n_basis == 0:
            self.w_vs.data.copy_(w_vs.view(k, h_size, h_size, n_head).permute(0, 3, 1, 2))
        else:
            self.w_vs.data.copy_(w_vs.permute(0, 2, 1))
            self.w_vs_co.data.copy_(w_vs_co.permute(0, 2, 1))

    def _get_weights(self):
        if self.diag_decompose:
            W, Wi = self.w_vs[:, :, :-1], self.w_vs[:, :, -1]
        elif self.n_basis == 0:
            W, Wi = self.w_vs[:, :-1, :, :], self.w_vs[:, -1, :, :]
        else:
            W = self.w_vs_co.bmm(self.w_vs).view(self.k, self.n_head, self.hidden_size, self.hidden_size)
            W, Wi = W[:, :-1, :, :], W[:, -1, :, :]

        k, h_size = self.k, self.hidden_size
        W_pad = [W.new_ones((h_size,)) if self.diag_decompose else torch.eye(h_size, device=W.device)]
        for t in range(k - 1):
            if self.diag_decompose:
                W_pad = [Wi[k - 1 - t] * W_pad[0]] + W_pad
            else:
                W_pad = [Wi[k - 1 - t].mm(W_pad[0])] + W_pad
        assert len(W_pad) == k
        return W, W_pad

    def decode(self, end_ids, ks, A, start_attn, uni_attn, trans_attn):
        """
        end_ids: tensor of shape (batch_size,)
        ks: tensor of shape (batch_size,)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)

        returns: list[tensor of shape (path_len,)]
        """
        bs, n_head, n_node, n_node = A.size()
        assert ((A == 0) | (A == 1)).all()

        path_ids = end_ids.new_zeros((bs, self.k * 2 + 1))
        path_lengths = end_ids.new_zeros((bs,))

        for idx in range(bs):
            back_trace = []
            end_id, k, adj = end_ids[idx], ks[idx], A[idx]
            uni_a, trans_a, start_a = uni_attn[idx], trans_attn[idx], start_attn[idx]

            if (adj[:, end_id, :] == 0).all():  # end_id is not connected to any other node
                path_ids[idx, 0] = end_id
                path_lengths[idx] = 1
                continue

            dp = F.one_hot(end_id, num_classes=n_node).float()  # (n_node,)
            assert 1 <= k <= self.k
            for t in range(k):
                if t == 0:
                    dp = dp.unsqueeze(0).expand(n_head, n_node)
                else:
                    dp = dp.unsqueeze(0) * trans_a.unsqueeze(-1)  # (n_head, n_head, n_node)
                    dp, ptr = dp.max(1)
                    back_trace.append(ptr)  # (n_head, n_node)
                dp = dp.unsqueeze(-1) * adj  # (n_head, n_node, n_node)
                dp, ptr = dp.max(1)
                back_trace.append(ptr)  # (n_head, n_node)
                dp = dp * uni_a.unsqueeze(-1)  # (n_head, n_node)
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # (n_node,)
            dp = dp * start_a
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # （)
            assert dp.dim() == 0
            assert len(back_trace) == k + (k - 1) + 2

            # re-construct path from back_trace
            path = end_id.new_zeros((2 * k + 1,))  # (k + 1) entities and k relations
            path[0] = back_trace.pop(-1)
            path[1] = back_trace.pop(-1)[path[0]]
            for p in range(2, 2 * k + 1):
                if p % 2 == 0:  # need to fill a entity id
                    path[p] = back_trace.pop(-1)[path[p - 1], path[p - 2]]
                else:  # need to fill a relation id
                    path[p] = back_trace.pop(-1)[path[p - 2], path[p - 1]]
            assert len(back_trace) == 0
            assert path[-1] == end_id
            path_ids[idx, :2 * k + 1] = path
            path_lengths[idx] = 2 * k + 1

        return path_ids, path_lengths

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        W, W_pad = self._get_weights()  # (k, h_size, n_head) or (k, n_head, h_size h_size)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)

        Z_all = []
        Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
        for t in range(k):
            if t == 0:  # Z.size() == (bs, n_node, h_size)
                Z = Z.unsqueeze(-1).expand(bs, n_node, h_size, n_head)
            else:  # Z.size() == (bs, n_head, n_node, h_size)
                Z = Z.permute(0, 2, 3, 1).view(bs, n_node * h_size, n_head)
                Z = Z.bmm(trans_attn).view(bs, n_node, h_size, n_head)
            if self.diag_decompose:
                Z = Z * W[t]  # (bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
            else:
                Z = Z.permute(3, 0, 1, 2).view(n_head, bs * n_node, h_size)
                Z = Z.bmm(W[t]).view(n_head, bs, n_node, h_size)
                Z = Z.permute(1, 0, 2, 3).contiguous().view(bs * n_head, n_node, h_size)
            Z = Z * uni_attn[:, None, None]
            Z = A.bmm(Z)
            Z = Z.view(bs, n_head, n_node, h_size)
            Zt = Z.sum(1) * W_pad[t] if self.diag_decompose else Z.sum(1).matmul(W_pad[t])
            Zt = Zt * end_attn.unsqueeze(2)
            Z_all.append(Zt)

        # compute the normalization factor
        D_all = []
        D = start_attn
        for t in range(k):
            if t == 0:  # D.size() == (bs, n_node)
                D = D.unsqueeze(1).expand(bs, n_head, n_node)
            else:  # D.size() == (bs, n_head, n_node)
                D = D.permute(0, 2, 1).bmm(trans_attn)  # (bs, n_node, n_head)
                D = D.permute(0, 2, 1)
            D = D.contiguous().view(bs * n_head, n_node, 1)
            D = D * uni_attn[:, None, None]
            D = A.bmm(D)
            D = D.view(bs, n_head, n_node)
            Dt = D.sum(1) * end_attn
            D_all.append(Dt)

        Z_all = [Z / (D.unsqueeze(2) + self.eps) for Z, D in zip(Z_all, D_all)]
        assert len(Z_all) == k
        if 'agg_self_loop' in self.ablation:
            Z_all = [X] + Z_all
        return Z_all


class PathAttentionLayer(nn.Module):
    def __init__(self, n_type, n_head, sent_dim, att_dim, att_layer_num, dropout=0.1, ablation=[]):
        super().__init__()
        self.n_head = n_head
        self.ablation = ablation
        if 'no_att' not in self.ablation:
            if 'no_type_att' not in self.ablation:
                self.start_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)
                self.end_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)

            if 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
                self.path_uni_attention = MLP(sent_dim, att_dim, n_head, att_layer_num, dropout, layer_norm=True)

            if 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
                if 'ctx_trans' in self.ablation:
                    self.path_pair_attention = MLP(sent_dim, att_dim, n_head ** 2, 1, dropout, layer_norm=True)
                self.trans_scores = nn.Parameter(torch.zeros(n_head ** 2))

    def forward(self, S, node_type):
        """
        S: tensor of shape (batch_size, d_sent)
        node_type: tensor of shape (batch_size, n_node)

        returns: tensors of shapes (batch_size, n_node) (batch_size, n_node) (batch_size, n_head) (batch_size, n_head, n_head)
        """
        n_head = self.n_head
        bs, n_node = node_type.size()

        if 'detach_s_all' in self.ablation:
            S = S.detach()

        if 'no_att' not in self.ablation and 'no_type_att' not in self.ablation:
            bi = torch.arange(bs).unsqueeze(-1).expand(bs, n_node).contiguous().view(-1)  # [0 ... 0 1 ... 1 ...]
            start_attn = self.start_attention(S)
            if 'q2a_only' in self.ablation:
                start_attn[:, 1] = -np.inf
                start_attn[:, 2] = -np.inf
            start_attn = torch.exp(start_attn - start_attn.max(1, keepdim=True)[0])  # softmax trick to avoid numeric overflow
            start_attn = start_attn[bi, node_type.view(-1)].view(bs, n_node)
            end_attn = self.end_attention(S)
            if 'q2a_only' in self.ablation:
                end_attn[:, 0] = -np.inf
                end_attn[:, 2] = -np.inf
            end_attn = torch.exp(end_attn - end_attn.max(1, keepdim=True)[0])
            end_attn = end_attn[bi, node_type.view(-1)].view(bs, n_node)
        else:
            start_attn = torch.ones((bs, n_node), device=S.device)
            end_attn = torch.ones((bs, n_node), device=S.device)

        if 'no_att' not in self.ablation and 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
            uni_attn = self.path_uni_attention(S).view(bs, n_head)  # (bs, n_head)
            uni_attn = torch.exp(uni_attn - uni_attn.max(1, keepdim=True)[0]).view(bs, n_head)
        else:
            uni_attn = torch.ones((bs, n_head), device=S.device)

        if 'no_att' not in self.ablation and 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
            if 'ctx_trans' in self.ablation:
                trans_attn = self.path_pair_attention(S) + self.trans_scores
            else:
                trans_attn = self.trans_scores.unsqueeze(0).expand(bs, n_head ** 2)
            trans_attn = torch.exp(trans_attn - trans_attn.max(1, keepdim=True)[0])
            trans_attn = trans_attn.view(bs, n_head, n_head)
        else:
            trans_attn = torch.ones((bs, n_head, n_head), device=S.device)
        return start_attn, end_attn, uni_attn, trans_attn


class Aggregator(nn.Module):

    def __init__(self, sent_dim, hidden_size, ablation=[]):
        super().__init__()
        self.ablation = ablation
        self.w_qs = nn.Linear(sent_dim, hidden_size)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (sent_dim + hidden_size)))
        self.temperature = np.power(hidden_size, 0.5)
        self.softmax = nn.Softmax(2)

    def forward(self, S, Z_all):
        """
        S: tensor of shape (batch_size, d_sent)
        Z_all: tensor of shape (batch_size, n_node, k, d_node)
        returns: tensor of shape (batch_size, n_node, d_node)
        """
        if 'detach_s_agg' in self.ablation or 'detach_s_all' in self.ablation:
            S = S.detach()
        S = self.w_qs(S)  # (bs, d_node)
        attn = (S[:, None, None, :] * Z_all).sum(-1)  # (bs, n_node, k)
        if 'no_1hop' in self.ablation:
            if 'agg_self_loop' in self.ablation:
                attn[:, :, 1] = -np.inf
            else:
                attn[:, :, 0] = -np.inf

        attn = self.softmax(attn / self.temperature)
        Z = (attn.unsqueeze(-1) * Z_all).sum(2)
        return Z, attn


class GraphRelationLayer(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, input_size, hidden_size, output_size, sent_dim,
                 att_dim, att_layer_num, dropout=0.1, diag_decompose=False, eps=1e-20, ablation=None):
        super().__init__()
        assert input_size == output_size
        self.ablation = ablation

        if 'no_typed_transform' not in self.ablation:
            self.typed_transform = TypedLinear(input_size, hidden_size, n_type)
        else:
            assert input_size == hidden_size

        self.path_attention = PathAttentionLayer(n_type, n_head, sent_dim, att_dim, att_layer_num, dropout, ablation=ablation)
        self.message_passing = MultiHopMessagePassingLayer(k, n_head, hidden_size, diag_decompose, n_basis, eps=eps, ablation=ablation)
        self.aggregator = Aggregator(sent_dim, hidden_size, ablation=ablation)

        self.Vh = nn.Linear(input_size, output_size)
        self.Vz = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def decode(self, end_ids, A):
        ks = self.len_attn.argmax(2)  # (bs, n_node)
        if 'detach_s_agg' not in self.ablation:
            ks = ks + 1
        ks = ks.gather(1, end_ids.unsqueeze(-1)).squeeze(-1)  # (bs,)
        path_ids, path_lenghts = self.message_passing.decode(end_ids, ks, A, self.start_attn, self.uni_attn, self.trans_attn)
        return path_ids, path_lenghts

    def forward(self, S, H, A, node_type, cache_output=False):
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

        if 'no_typed_transform' not in self.ablation:
            X = self.typed_transform(H, node_type)
        else:
            X = H

        start_attn, end_attn, uni_attn, trans_attn = self.path_attention(S, node_type)

        Z_all = self.message_passing(X, A, start_attn, end_attn, uni_attn, trans_attn)
        Z_all = torch.stack(Z_all, 2)  # (bs, n_node, k, h_size) or (bs, n_node, k+1, h_size)
        Z, len_attn = self.aggregator(S, Z_all)

        if cache_output:  # cache intermediate ouputs for decoding
            self.start_attn, self.uni_attn, self.trans_attn = start_attn, uni_attn, trans_attn
            self.len_attn = len_attn  # (bs, n_node, k)

        if 'early_relu' in self.ablation:
            output = self.Vh(H) + self.activation(self.Vz(Z))
        else:
            output = self.activation(self.Vh(H) + self.Vz(Z))

        output = self.dropout(output)
        return output


class GraphRelationEncoder(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, n_layer, input_size, hidden_size, sent_dim,
                 att_dim, att_layer_num, dropout, diag_decompose, eps=1e-20, ablation=None):
        super().__init__()
        self.layers = nn.ModuleList([GraphRelationLayer(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis,
                                                        input_size=input_size, hidden_size=hidden_size, output_size=input_size,
                                                        sent_dim=sent_dim, att_dim=att_dim, att_layer_num=att_layer_num,
                                                        dropout=dropout, diag_decompose=diag_decompose, eps=eps,
                                                        ablation=ablation) for _ in range(n_layer)])

    def decode(self, end_ids, A):
        bs = end_ids.size(0)
        k = self.layers[0].message_passing.k
        full_path_ids = end_ids.new_zeros((bs, k * 2 * len(self.layers) + 1))
        full_path_ids[:, 0] = end_ids
        full_path_lengths = end_ids.new_ones((bs,))
        for layer in self.layers[::-1]:
            path_ids, path_lengths = layer.decode(end_ids, A)
            for i in range(bs):
                prev_l = full_path_lengths[i]
                inc_l = path_lengths[i]
                path = path_ids[i]
                assert full_path_ids[i, prev_l - 1] == path[inc_l - 1]
                full_path_ids[i, prev_l:prev_l + inc_l - 1] = path_ids[i, :inc_l - 1].flip((0,))
                full_path_lengths[i] = prev_l + inc_l - 1
        for i in range(bs):
            full_path_ids[i, :full_path_lengths[i]] = full_path_ids[i, :full_path_lengths[i]].flip((0,))
        return full_path_ids, full_path_lengths

    def forward(self, S, H, A, node_type_ids, cache_output=False):
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
            H = layer(S, H, A, node_type_ids, cache_output=cache_output)
        return H


class GraphRelationNet(nn.Module):
    def __init__(self, k, n_type, n_basis, n_layer, sent_dim, diag_decompose,
                 n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True, ablation=None,
                 init_range=0.02, eps=1e-20, use_contextualized=False, do_init_rn=False, do_init_identity=False):
        super().__init__()
        self.ablation = ablation
        self.init_range = init_range
        self.do_init_rn = do_init_rn
        self.do_init_identity = do_init_identity

        n_head = 1 if 'no_rel' in self.ablation else n_relation

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)

        self.gnn = GraphRelationEncoder(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis, n_layer=n_layer,
                                        input_size=concept_dim, hidden_size=concept_dim, sent_dim=sent_dim,
                                        att_dim=att_dim, att_layer_num=att_layer_num, dropout=p_gnn,
                                        diag_decompose=diag_decompose, eps=eps, ablation=ablation)

        if 'early_trans' in self.ablation:
            self.typed_transform = TypedLinear(concept_dim, concept_dim, n_type=n_type)

        if 'typed_pool' in self.ablation and 'early_trans' not in self.ablation:
            self.pooler = TypedMultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim, n_type=n_type)
        else:
            self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_rn(self, module):
        if hasattr(module, 'typed_transform'):
            h_size = module.typed_transform.out_features
            half_h_size = h_size // 2
            bias = module.typed_transform.bias
            new_bias = bias.data.clone().detach().view(-1, h_size)
            new_bias[:, :half_h_size] = 1
            bias.data.copy_(new_bias.view(-1))

    def _init_identity(self, module):
        if module.diag_decompose:
            module.w_vs.data[:, :, -1] = 1
        elif module.n_basis == 0:
            module.w_vs.data[:, -1, :, :] = torch.eye(module.w_vs.size(-1), device=module.w_vs.device)
        else:
            print('Warning: init_identity not implemented for n_basis > 0')
            pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MultiHopMessagePassingLayer):
            if 'fix_scale' in self.ablation:
                module.w_vs.data.normal_(mean=0.0, std=np.sqrt(np.pi / 2))
            else:
                module.w_vs.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'w_vs_co'):
                getattr(module, 'w_vs_co').data.fill_(1.0)
            if self.do_init_identity:
                self._init_identity(module)
        elif isinstance(module, PathAttentionLayer):
            if hasattr(module, 'trans_scores'):
                getattr(module, 'trans_scores').data.zero_()
        elif isinstance(module, GraphRelationLayer) and self.do_init_rn:
            self._init_rn(module)

    def decode(self):
        bs, _, n_node, _ = self.adj.size()
        end_ids = self.pool_attn.view(-1, bs, n_node)[0, :, :].argmax(-1)  # use only the first head if multi-head attention
        path_ids, path_lengths = self.gnn.decode(end_ids, self.adj)

        # translate local entity ids (0~200) into global eneity ids (0~7e5)
        entity_ids = path_ids[:, ::2]  # (bs, ?)
        path_ids[:, ::2] = self.concept_ids.gather(1, entity_ids)
        return path_ids, path_lengths

    def forward(self, sent_vecs, concept_ids, node_type_ids, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node

        returns: (batch_size, 1)
        """
        gnn_input = self.dropout_e(self.concept_emb(concept_ids, emb_data))
        if 'no_ent' in self.ablation:
            gnn_input[:] = 1.0
        if 'no_rel' in self.ablation:
            adj = adj.sum(1, keepdim=True)
        gnn_output = self.gnn(sent_vecs, gnn_input, adj, node_type_ids, cache_output=cache_output)

        mask = torch.arange(concept_ids.size(1), device=adj.device) >= adj_lengths.unsqueeze(1)
        if 'pool_qc' in self.ablation:
            mask = mask | (node_type_ids != 0)
        elif 'pool_all' in self.ablation:
            mask = mask
        else:  # default is to perform pooling over all the answer concepts (pool_ac)
            mask = mask | (node_type_ids != 1)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        if 'early_trans' in self.ablation:
            gnn_output = self.typed_transform(gnn_output, type_ids=node_type_ids)

        if 'detach_s_pool' in self.ablation:
            sent_vecs_for_pooler = sent_vecs.detach()
        else:
            sent_vecs_for_pooler = sent_vecs

        if 'typed_pool' in self.ablation and 'early_trans' not in self.ablation:
            graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask, type_ids=node_type_ids)
        else:
            graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:  # cache for decoding
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn


class LMGraphRelationNet(nn.Module):
    def __init__(self, model_name, k, n_type, n_basis, n_layer, diag_decompose,
                 n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True, ablation=None,
                 init_range=0.0, eps=1e-20, use_contextualized=False,
                 do_init_rn=False, do_init_identity=False, encoder_config={}):
        super().__init__()
        self.ablation = ablation
        self.use_contextualized = use_contextualized
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = GraphRelationNet(k, n_type, n_basis, n_layer, self.encoder.sent_dim, diag_decompose,
                                        n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        ablation=ablation, init_range=init_range, eps=eps, use_contextualized=use_contextualized,
                                        do_init_rn=do_init_rn, do_init_identity=do_init_identity)

    def decode(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        logits, _ = self.forward(*inputs, layer_id=layer_id, cache_output=True)
        path_ids, path_lengths = self.decoder.decode()
        assert (path_lengths % 2 == 1).all()
        path_ids = path_ids.view(bs, nc, -1)
        path_lengths = path_lengths.view(bs, nc)
        return logits, path_ids, path_lengths

    def forward(self, *inputs, layer_id=-1, cache_output=False):
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
        if not self.use_contextualized:
            *lm_inputs, concept_ids, node_type_ids, adj_lengths, adj = inputs
            emb_data = None
        else:
            *lm_inputs, concept_ids, node_type_ids, adj_lengths, emb_data, adj = inputs
        if 'no_lm' not in self.ablation:
            sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        else:
            sent_vecs = torch.ones((bs * nc, self.encoder.sent_dim), dtype=torch.float)
        logits, attn = self.decoder(sent_vecs.to(concept_ids.device), concept_ids, node_type_ids, adj_lengths, adj,
                                    emb_data=emb_data, cache_output=cache_output)
        logits = logits.view(bs, nc)
        return logits, attn


class LMGraphRelationNetDataLoader(object):

    def __init__(self, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 train_embs_path=None, dev_embs_path=None, test_embs_path=None,
                 is_inhouse=False, inhouse_train_qids_path=None, use_contextualized=False,
                 subsample=1.0, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.use_contextualized = use_contextualized

        model_type = MODEL_NAME_TO_CLASS[model_name]
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)

        num_choice = self.train_encoder_data[0].size(1)
        *self.train_decoder_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, num_choice, emb_pk_path=train_embs_path if use_contextualized else None)
        *self.dev_decoder_data, self.dev_adj_data, n_rel = load_adj_data(dev_adj_path, max_node_num, num_choice, emb_pk_path=dev_embs_path if use_contextualized else None)
        assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        # pre-allocate an empty batch adj matrix
        self.adj_empty = torch.zeros((self.batch_size, num_choice, n_rel - 1, max_node_num, max_node_num), dtype=torch.float32)
        self.eval_adj_empty = torch.zeros((self.eval_batch_size, num_choice, n_rel - 1, max_node_num, max_node_num), dtype=torch.float32)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
            *self.test_decoder_data, self.test_adj_data, n_rel = load_adj_data(test_adj_path, max_node_num, num_choice, emb_pk_path=test_embs_path if use_contextualized else None)
            assert all(len(self.test_qids) == len(self.test_adj_data) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

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
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def get_node_feature_dim(self):
        return self.train_decoder_data[-1].size(-1) if self.use_contextualized else None

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


def run_test():
    import time
    print('***** testing MultiHopMessagePassingLayer *****')
    for diag_decompose in (True, False):
        for k in range(1, 4):
            print(f'running k = {k}')
            n_samples, n_node, n_head, h_size = 100, 200, 34, 100
            model0 = MultiHopMessagePassingLayerDeprecated(k, n_head, h_size, diag_decompose, 0, ablation=['no_trans'])
            model1 = MultiHopMessagePassingLayerDeprecated(k, n_head, h_size, diag_decompose, 0)
            model2 = MultiHopMessagePassingLayer(k, n_head, h_size, diag_decompose, 0)
            model2.init_from_old(model1.w_vs.data)
            X = torch.randn(n_samples, n_node, h_size)
            A = (torch.randn(n_samples, n_head, n_node, n_node) > 0).float()
            start_attn = torch.rand(n_samples, n_node)
            end_attn = torch.rand(n_samples, n_node)
            uni_attn = torch.rand(n_samples, n_head)
            trans_attn = torch.rand(n_samples, n_head * n_head)
            start_attn, end_attn, uni_attn, trans_attn = [torch.exp(x - x.max(-1, keepdim=True)[0]) for x in (start_attn, end_attn, uni_attn, trans_attn)]
            uni_attn = uni_attn.view(n_samples, n_head)
            trans_attn = trans_attn.view(n_samples, n_head, n_head)
            model0.eval()
            model1.eval()
            model2.eval()
            res = []
            times = []
            for model in (model0, model1, model2):
                res.append([])
                times.append(time.time())
                with torch.no_grad():
                    bs = 2
                    for a in range(0, n_samples, bs):
                        b = min(a + bs, n_samples)
                        inputs = [x[a:b] for x in (X, A, start_attn, end_attn, uni_attn, trans_attn)]
                        Z = model(*inputs)
                        res[-1].append(Z)
                times[-1] = (time.time() - times[-1]) / (n_samples / bs)
            print(('|' + ' {:.2f} ms/batch |' * len(times)).format(*[t * 1000 for t in times]))
            for Zs in zip(*res):
                Zs = Zs[1:]
                try:
                    assert all(((Z - Zs[0]).abs() < 1e-6).all() for Z in Zs[1:])
                except AssertionError:
                    print(Zs)
                    raise
    print('***** all tests are passed *****')
