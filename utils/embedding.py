import numpy as np
from tqdm import tqdm
from utils.tokenization_utils import EXTRA_TOKS

__all__ = ['glove2npy', 'load_vectors_from_npy_with_vocab', ]


def load_vectors(path, skip_head=False, add_special_tokens=None, random_state=0):
    vocab = []
    vectors = None
    nrow = sum(1 for line in open(path, 'r', encoding='utf-8'))
    with open(path, "r", encoding="utf8") as fin:
        if skip_head:
            fin.readline()
        for i, line in tqdm(enumerate(fin), total=nrow):
            elements = line.strip().split(" ")
            word = elements[0].lower()
            vec = np.array(elements[1:], dtype=float)
            vocab.append(word)
            if vectors is None:
                vectors = np.zeros((nrow, len(vec)), dtype=np.float64)
            vectors[i] = vec

    np.random.seed(random_state)
    n_special = 0 if add_special_tokens is None else len(add_special_tokens)
    add_vectors = np.random.normal(np.mean(vectors), np.std(vectors), size=(n_special, vectors.shape[1]))
    vectors = np.concatenate((vectors, add_vectors), 0)
    vocab += add_special_tokens
    return vocab, vectors


def glove2npy(glove_path, output_npy_path, output_vocab_path, skip_head=False,
              add_special_tokens=EXTRA_TOKS, random_state=0):
    print('binarizing GloVe embeddings...')

    vocab, vectors = load_vectors(glove_path, skip_head=skip_head,
                                  add_special_tokens=add_special_tokens, random_state=random_state)
    np.save(output_npy_path, vectors)
    with open(output_vocab_path, "w", encoding='utf-8') as fout:
        for word in vocab:
            fout.write(word + '\n')

    print(f'Binarized GloVe embeddings saved to {output_npy_path}')
    print(f'GloVe vocab saved to {output_vocab_path}')
    print()


def load_vectors_from_npy_with_vocab(glove_npy_path, glove_vocab_path, vocab, verbose=True, save_path=None):
    with open(glove_vocab_path, 'r') as fin:
        glove_w2idx = {line.strip(): i for i, line in enumerate(fin)}
    glove_emb = np.load(glove_npy_path)
    vectors = np.zeros((len(vocab), glove_emb.shape[1]), dtype=float)
    oov_cnt = 0
    for i, word in enumerate(vocab):
        if word in glove_w2idx:
            vectors[i] = glove_emb[glove_w2idx[word]]
        else:
            oov_cnt += 1
    if verbose:
        print(len(vocab))
        print('embedding oov rate: {:.4f}'.format(oov_cnt / len(vocab)))
    if save_path is None:
        return vectors
    np.save(save_path, vectors)


def load_pretrained_embeddings(glove_npy_path, glove_vocab_path, vocab_path, verbose=True, save_path=None):
    vocab = []
    with open(vocab_path, 'r') as fin:
        for line in fin.readlines():
            vocab.append(line.strip())
    load_vectors_from_npy_with_vocab(glove_npy_path=glove_npy_path, glove_vocab_path=glove_vocab_path, vocab=vocab, verbose=verbose, save_path=save_path)


if __name__ == "__main__":
    glove2npy(glove_path='../data/transe/numberbatch-en-19.08.txt', output_npy_path='../data/transe/nb.npy', output_vocab_path='../data/transe/nb.vocab', skip_head=True)
    load_pretrained_embeddings(glove_npy_path='../data/transe/nb.npy', glove_vocab_path='../data/transe/nb.vocab', vocab_path='../data/cpnet/concept.txt', save_path='../data/transe/concept.nb.npy')
