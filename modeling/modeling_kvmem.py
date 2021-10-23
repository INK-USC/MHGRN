from modeling.modeling_lm import *
from utils.data_utils import *
from utils.layers import *


class KVM(nn.Module):
    def __init__(self, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, relation_num, s_dim, num_layers, bidirectional,
                 input_p, output_p, emb_p, hidden_p, dropoutm, mask_with_s_len, gamma=0.5):
        super().__init__()

        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, use_contextualized=False)
        self.relation_emb = CustomizedEmbedding(concept_num=relation_num, concept_out_dim=concept_dim, concept_in_dim=concept_dim,
                                                pretrained_concept_emb=None, freeze_ent_emb=False, use_contextualized=False)
        self.hidden_dim = concept_dim
        self.s_dim = s_dim
        self.gamma = gamma
        self.mask_with_s_len = mask_with_s_len


        if self.s_dim != self.hidden_dim:
            print(f'sentence dim: {self.s_dim}')
            print(f'triple repr dim: {self.hidden_dim}')
            print('The dimension of token representation and output of triple encoder doesn\'t match, linear transformation applied.')
            self.transform = nn.Linear(self.s_dim, self.hidden_dim)

        self.sim = DotProductSimilarity()
        self.attention = MatrixAttention(self.sim)
        self.triple_encoder = TripleEncoder(emb_dim=concept_dim, hidden_dim=self.hidden_dim, input_p=input_p, output_p=output_p,
                                            hidden_p=hidden_p, num_layers=num_layers, bidirectional=bidirectional,
                                            pad=False, concept_emb=self.concept_emb, relation_emb=self.relation_emb)
        # self.MLP = MLP(input_size=4 * concept_dim, hidden_size=hidden_dim, output_size=hidden_dim, num_layers=1, dropout=dropoutm)
        # self.hidd2out = MLP(input_size=4*hidden_dim+s_dim, hidden_size=hidden_dim, output_size=1, num_layers=1, dropout=dropout)
        self.hidd2out = nn.Linear(self.hidden_dim, 1)
        self.max_pool = MaxPoolLayer()
        # self.mean_pool = MeanPoolLayer()

    def forward(self, t, t_num, s, s_mask):
        """
        t: (nbz, 3 * num_padded_triples)
        t_num: (nbz, )
        s: (nbz, max_token_num, token_hidden_dim)
        if self.mask_with_s_len:
            s_mask: (nbz, )
        else:
            s_mask: (nbz, max_token_num)
        """

        bz, t_sl = t.size()
        t_sl = t_sl // 3
        _, s_sl, _ = s.size()

        t_num = torch.max(t_num, torch.tensor(1).to(t_num.device)).unsqueeze(1)
        t_mask = torch.arange(t_sl, device=t.device) >= t_num  # (nbz, 1)
        if self.mask_with_s_len:
            s_mask = torch.max(s_mask, torch.tensor(1).to(s_mask.device)).unsqueeze(1)
            s_mask = torch.arange(s_sl, device=t.device) >= s_mask  # (nbz, 1)

        mask = s_mask.unsqueeze(2) | t_mask.unsqueeze(1)  # (nbz, s_sl, t_sl)
        t_repr = self.triple_encoder(t.view(bz * t_sl, 3)).view(bz, t_sl, -1)  # (nbz, t_sl, h_dim)
        if self.s_dim != self.hidden_dim:
            s = self.transform(s)

        attn = self.attention(s, t_repr)  # (nbz, s_sl, t_sl)

        s2t = masked_softmax(attn, mask, dim=-1)  # (nbz, s_sl, t_sl)

        beta = (s2t.unsqueeze(3) * t_repr.unsqueeze(1)).sum(2)  # (nbz, s_sl, h_dim), unsqueeze dim of t, sum over dim of t
        # if self.s_dim != self.hidden_dim:
        #     beta = self.transform(beta)

        # hidden = self.mean_pool((self.gamma * beta + (1 - self.gamma) * s), s_len.squeeze(1))
        hidden = self.max_pool((self.gamma * beta + (1 - self.gamma) * s), s_mask)
        logits = self.hidd2out(hidden)
        return logits


class LMKVM(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, concept_emb, relation_num,
                 decoder_num_layers, decoder_bidirectional, decoder_input_p, decoder_output_p, decoder_emb_p, decoder_hidden_p,
                 decoder_mlp_p, gamma, encoder_config={}):
        super().__init__()
        self.model_name = model_name
        mask_with_s_len = model_name in ('lstm',)
        self.encoder = TextEncoder(model_name, output_token_states=True, **encoder_config)
        self.decoder = KVM(concept_num=concept_num, concept_dim=concept_dim, concept_in_dim=concept_in_dim,
                           freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=concept_emb,
                           relation_num=relation_num,
                           s_dim=self.encoder.sent_dim, num_layers=decoder_num_layers, bidirectional=decoder_bidirectional,
                           input_p=decoder_input_p, output_p=decoder_output_p, emb_p=decoder_emb_p, hidden_p=decoder_hidden_p,
                           dropoutm=decoder_mlp_p, gamma=gamma, mask_with_s_len=mask_with_s_len)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        if self.model_name in ('lstm',):
            statement_data, statement_len, triples, triple_num = inputs
            hidden_states = self.encoder(statement_data, statement_len)
            logits = self.decoder(t=triples, t_num=triple_num, s=hidden_states.to(triple_num.device), s_mask=statement_len.to(triple_num.device))
        else:  # bert / xlnet / roberta  TODO: support GPT
            *lm_inputs, triples, triple_num = inputs
            hidden_states, output_mask = self.encoder(*lm_inputs, layer_id=layer_id)
            logits = self.decoder(t=triples, t_num=triple_num, s=hidden_states.to(triple_num.device), s_mask=output_mask.to(triple_num.device))
        logits = logits.view(bs, nc)

        return logits, None  # for lstm encoder, attn=None


class KVMDataLoader(object):
    def __init__(self, train_statement_path: str, train_triple_pk: str, dev_statement_path: str,
                 dev_triple_pk: str, test_statement_path: str, test_triple_pk: str,
                 concept2id_path: str, batch_size, eval_batch_size, device, model_name=None,
                 max_triple_num=200, max_seq_length=128, is_inhouse=True, inhouse_train_qids_path=None,
                 subsample=1.0, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse
        self.max_triple_num = max_triple_num
        self.vocab = None
        model_type = MODEL_NAME_TO_CLASS.get(model_name, 'lstm')

        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)
        self.num_choice = self.train_encoder_data[0].size(1)

        self._load_concept_idx(concept2id_path)

        self.train_decoder_data = self._load_triples(train_triple_pk)
        self.dev_decoder_data = self._load_triples(dev_triple_pk)
        if test_statement_path is not None:
            self.test_decoder_data = self._load_triples(test_triple_pk)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + list(self.train_decoder_data))
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + list(self.dev_decoder_data))
        if test_statement_path is not None:
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + list(self.test_decoder_data))
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
                assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def __getitem__(self, index):
        raise NotImplementedError()

    def _load_concept_idx(self, concept_list):
        with open(concept_list, 'r', encoding='utf8') as fin:
            id2concept = [w.strip() for w in fin]

        self.concept2id = {w: i for i, w in enumerate(id2concept)}

    def _load_triples(self, triple_path):
        with open(triple_path, 'rb') as fin:
            triples, mc_triple_num = pickle.load(fin)
        t = torch.full((len(triples), self.max_triple_num * 3), 0, dtype=torch.int64)
        triple_num = []
        for idx, (i, j, k) in tqdm(enumerate(triples), total=len(triples), desc='loading triples'):
            i, j, k = [torch.tensor(x[:self.max_triple_num]) for x in [i, j, k]]
            if len(i) > 0:
                t[idx][:(len(i) * 3)] = torch.cat((j.unsqueeze(1), i.unsqueeze(1), k.unsqueeze(1)), dim=1).view(-1)
            triple_num.append(len(i))
        triple_num = torch.tensor(triple_num, dtype=torch.int64).view(-1, self.num_choice)

        return t.view(-1, self.num_choice, self.max_triple_num * 3), triple_num

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        return self.inhouse_test_indexes.size(0) if self.is_inhouse else len(self.test_qids)

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUBatchGenerator(self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data,
                                      tensors1=self.train_decoder_data)

    def dev(self):
        return MultiGPUBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data,
                                      tensors1=self.dev_decoder_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUBatchGenerator(self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data,
                                          tensors1=self.train_decoder_data)
        else:
            return MultiGPUBatchGenerator(self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data,
                                          tensors1=self.test_decoder_data)
