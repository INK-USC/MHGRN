import nltk
import json
from tqdm import tqdm


__all__ = ['WordVocab', 'EOS_TOK', 'UNK_TOK', 'PAD_TOK', 'EXTRA_TOKS', 'tokenize_statement_file']


EOS_TOK = '<EOS>'
UNK_TOK = '<UNK>'
PAD_TOK = '<PAD>'
EXTRA_TOKS = [EOS_TOK, UNK_TOK, PAD_TOK]


class WordVocab(object):

    def __init__(self, sents=None, path=None, freq_cutoff=5, encoding='utf-8', verbose=True):
        """
        sents: list[str] (optional, default None)
        path: str (optional, default None)
        freq_cutoff: int (optional, default 5, 0 to disable)
        encoding: str (optional, default utf-8)
        """
        if sents is not None:
            counts = {}
            for text in sents:
                for w in text.split():
                    counts[w] = counts.get(w, 0) + 1
            self._idx2w = [t[0] for t in sorted(counts.items(), key=lambda x: -x[1])]
            self._w2idx = {w: i for i, w in enumerate(self._idx2w)}
            self._counts = counts

        elif path is not None:
            self._idx2w = []
            self._counts = {}
            with open(path, 'r', encoding=encoding) as fin:
                for line in fin:
                    w, c = line.rstrip().split(' ')
                    self._idx2w.append(w)
                    self._counts[w] = c
                self._w2idx = {w: i for i, w in enumerate(self._idx2w)}

        else:
            self._idx2w = []
            self._w2idx = {}
            self._counts = {}

        if freq_cutoff > 1:
            self._idx2w = [w for w in self._idx2w if self._counts[w] >= freq_cutoff]

            in_sum = sum([self._counts[w] for w in self._idx2w])
            total_sum = sum([self._counts[w] for w in self._counts])
            if verbose:
                print('vocab oov rate: {:.4f}'.format(1 - in_sum / total_sum))

            self._w2idx = {w: i for i, w in enumerate(self._idx2w)}
            self._counts = {w: self._counts[w] for w in self._idx2w}

    def add_word(self, w, count=1):
        if w not in self.w2idx:
            self._w2idx[w] = len(self._idx2w)
            self._idx2w.append(w)
            self._counts[w] = count
        else:
            self._counts[w] += count
        return self

    def top_k_cutoff(self, size):
        if size < len(self._idx2w):
            for w in self._idx2w[size:]:
                self._w2idx.pop(w)
                self._counts.pop(w)
            self._idx2w = self._idx2w[:size]

        assert len(self._idx2w) == len(self._w2idx) == len(self._counts)
        return self

    def save(self, path, encoding='utf-8'):
        with open(path, 'w', encoding=encoding) as fout:
            for w in self._idx2w:
                fout.write(w + ' ' + str(self._counts[w]) + '\n')

    def __len__(self):
        return len(self._idx2w)

    def __contains__(self, word):
        return word in self._w2idx

    def __iter__(self):
        for word in self._idx2w:
            yield word

    @property
    def w2idx(self):
        return self._w2idx

    @property
    def idx2w(self):
        return self._idx2w

    @property
    def counts(self):
        return self._counts


def tokenize_sentence(sent, lower_case=True, convert_num=True):
    tokens = nltk.word_tokenize(sent)
    tokens = [t.lower() for t in tokens]
    tokens = ['<NUM>' if t.isdigit() else t for t in tokens]
    return tokens


def tokenize_statement_file(statement_path, output_path, lower_case=True, convert_num=True):
    nltk.download('punkt', quiet=True)
    nrow = sum(1 for _ in open(statement_path, 'r'))
    with open(statement_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, total=nrow):
            data = json.loads(line)
            for statement in data['statements']:
                tokens = tokenize_sentence(statement['statement'])
                fout.write(' '.join(tokens) + '\n')
