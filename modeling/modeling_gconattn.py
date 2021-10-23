from modeling.modeling_lm import *
from utils.data_utils import *
from utils.layers import *


class GconAttn(nn.Module):
    def __init__(self, concept_num, concept_dim, concept_in_dim, pretrained_concept_emb, freeze_ent_emb, hidden_dim, sent_dim, dropout):
        super().__init__()
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, use_contextualized=False)
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim

        self.sim = DotProductSimilarity()
        self.attention = MatrixAttention(self.sim)
        self.MLP = MLP(input_size=4 * concept_dim, hidden_size=hidden_dim, output_size=hidden_dim, num_layers=1, dropout=dropout)
        # self.hidd2out = MLP(input_size=4*hidden_dim+sent_dim, hidden_size=hidden_dim, output_size=1, num_layers=1, dropout=dropout)
        self.hidd2out = nn.Linear(4 * hidden_dim + sent_dim, 1)
        self.max_pool = MaxPoolLayer()
        self.mean_pool = MeanPoolLayer()

    def forward(self, q_id, a_id, q_num, a_num, s):
        """
        q_id: (nbz, seq_len)
        a_id: (nbz, seq_len)
        q_num: (nbz,)
        a_num: (nbz,)
        s: (nbz, sent_dim)
        """

        bz, sl = q_id.size()
        q_num = torch.max(q_num, torch.tensor(1).to(q_num.device)).unsqueeze(1)
        a_num = torch.max(a_num, torch.tensor(1).to(q_num.device)).unsqueeze(1)

        qmask = torch.arange(sl, device=q_id.device) >= q_num  # (nbz, sl)
        amask = torch.arange(sl, device=a_id.device) >= a_num  # (nbz, sl)

        mask = qmask.unsqueeze(2) | amask.unsqueeze(1)  # (nbz, sl, sl)

        q = self.concept_emb(q_id)  # (nbz, sl, cpt_dim)
        a = self.concept_emb(a_id)  # (nbz, sl, cpt_dim)
        attn = self.attention(q, a)  # (nbz, sl, sl)

        q2a = masked_softmax(attn, mask, dim=-1)  # (nbz, sl, sl)
        a2q = masked_softmax(attn, mask, dim=0)  # (nbz, sl, sl)

        beta = (q2a.unsqueeze(3) * a.unsqueeze(1)).sum(2)  # (nbz, sl, cpt_dim), unsqueeze dim of a, sum over dim of a
        alpha = (a2q.unsqueeze(3) * q.unsqueeze(2)).sum(1)  # (nbz, sl, cpt_dim), unsqueeze dim of q, sum over dim of q

        qm = self.MLP(torch.cat((a, beta, a - beta, a * beta), dim=-1))  # (nbz, sl, out_dim)
        am = self.MLP(torch.cat((q, alpha, q - alpha, q * alpha), dim=-1))  # (nbz, sl, out_dim)

        q_mean = self.mean_pool(qm, q_num.squeeze(1))  # (nbz, out_dim)
        q_max = self.max_pool(qm, q_num.squeeze(1))
        a_mean = self.mean_pool(am, a_num.squeeze(1))
        a_max = self.max_pool(am, a_num.squeeze(1))

        logits = self.hidd2out(torch.cat((q_mean, q_max, a_mean, a_max, s), dim=-1))

        return logits, None


class LMGconAttn(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}):
        super().__init__()
        self.model_name = model_name
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = GconAttn(concept_num=concept_num, concept_dim=concept_dim, concept_in_dim=concept_in_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=pretrained_concept_emb,
                                hidden_dim=hidden_dim, sent_dim=self.encoder.sent_dim, dropout=dropout)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        *lm_inputs, qc, ac, qc_len, ac_len = inputs
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)

        logits, attn = self.decoder(q_id=qc, a_id=ac, q_num=qc_len, a_num=ac_len, s=sent_vecs)
        logits = logits.view(bs, nc)
        return logits, attn  # for lstm encoder, attn=None


class GconAttnDataLoader(object):
    def __init__(self, train_statement_path: str, train_concept_jsonl: str, dev_statement_path: str,
                 dev_concept_jsonl: str, test_statement_path: str, test_concept_jsonl: str,
                 concept2id_path: str, batch_size, eval_batch_size, device, model_name=None,
                 max_cpt_num=20, max_seq_length=128, is_inhouse=True, inhouse_train_qids_path=None,
                 subsample=1.0, format=[]):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.max_cpt_num = max_cpt_num
        self.vocab = None

        model_type = MODEL_NAME_TO_CLASS.get(model_name, 'lstm')
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length, format=format)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length, format=format)
        self.num_choice = None
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length, format=format)

        self.num_choice = self.train_data[0].size(1)
        self._load_concept_idx(concept2id_path)

        self.train_data += self._load_concepts(train_concept_jsonl)
        self.dev_data += self._load_concepts(dev_concept_jsonl)
        if test_statement_path is not None:
            self.test_data += self._load_concepts(test_concept_jsonl)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r', encoding='utf-8') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        if test_statement_path is not None:
            assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)
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

    def __getitem__(self, index):
        raise NotImplementedError()

    def _load_concept_idx(self, concept_list):
        with open(concept_list, 'r', encoding='utf8') as fin:
            id2concept = [w.strip() for w in fin]

        self.concept2id = {w: i for i, w in enumerate(id2concept)}

    def _load_concepts(self, concept_json):

        with open(concept_json, 'r', encoding='utf-8') as fin:
            concept_data = [json.loads(line) for line in fin]
        n = len(concept_data)
        qc = []
        ac = []
        qc_len, ac_len = [], []
        for data in tqdm(concept_data, total=n, desc='loading concepts'):
            cur_qc = [self.concept2id[x] for x in data['qc']][:self.max_cpt_num]
            cur_ac = [self.concept2id[x] for x in data['ac']][:self.max_cpt_num]
            qc.append(cur_qc + [0] * (self.max_cpt_num - len(cur_qc)))
            ac.append(cur_ac + [0] * (self.max_cpt_num - len(cur_ac)))
            assert len(qc[-1]) == len(ac[-1]) == self.max_cpt_num
            qc_len.append(len(cur_qc))
            ac_len.append(len(cur_ac))

        print('avg_num_qc = {}'.format(sum(qc_len) / float(len(qc_len))))
        print('avg_num_ac = {}'.format(sum(ac_len) / float(len(ac_len))))
        qc, ac = [torch.tensor(np.array(x).reshape((-1, self.num_choice, self.max_cpt_num))) for x in [qc, ac]]
        qc_len, ac_len = [torch.tensor(np.array(x).reshape((-1, self.num_choice))) for x in [qc_len, ac_len]]
        return qc, ac, qc_len, ac_len

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        return self.inhouse_test_indexes.size(0) if self.is_inhouse else len(self.test_qids)

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors=self.train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)
