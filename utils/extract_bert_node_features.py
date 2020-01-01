import torch
import pickle
from tqdm import tqdm
import json
from transformers import BertModel, BertTokenizer
import argparse
import os

try:
    from .utils import check_path
except:
    from utils import check_path

id2concept = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]


def convert_qa_concept_to_bert_input(tokenizer, question, answer, concept, max_seq_length):
    qa_tokens = tokenizer.tokenize('Q: ' + question + ' A: ' + answer)
    concept_tokens = tokenizer.tokenize(concept)
    qa_tokens = qa_tokens[:max_seq_length - len(concept_tokens) - 3]
    tokens = [tokenizer.cls_token] + qa_tokens + [tokenizer.sep_token] + concept_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(qa_tokens) + 2) + [1] * (len(concept_tokens) + 1)

    assert len(input_ids) == len(segment_ids) == len(input_ids)

    # padding
    pad_len = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids += [0] * pad_len
    segment_ids += [0] * pad_len
    span = (len(qa_tokens) + 2, len(qa_tokens) + 2 + len(concept_tokens))

    assert span[1] + 1 == len(tokens)
    assert max_seq_length == len(input_ids) == len(segment_ids) == len(input_mask)

    return input_ids, input_mask, segment_ids, span


def extract_bert_node_features_from_adj(cpnet_vocab_path, statement_path, adj_path, output_path, max_seq_length, device, batch_size, layer_id=-1, cache_path=None, use_cache=True):
    global id2concept
    if id2concept is None:
        load_resources(cpnet_vocab_path=cpnet_vocab_path)
    check_path(output_path)

    print('extracting from triple strings')

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(device)
    model.eval()

    if use_cache and os.path.isfile(cache_path):
        print('Loading cached inputs.')
        with open(cache_path, 'rb') as fin:
            all_input_ids, all_input_mask, all_segment_ids, all_span, offsets = pickle.load(fin)
        print('Loaded')
    else:

        with open(adj_path, 'rb') as fin:
            adj_data = pickle.load(fin)

        offsets = [0]
        all_input_ids, all_input_mask, all_segment_ids, all_span = [], [], [], []

        n = sum(1 for _ in open(statement_path, 'r'))
        with open(statement_path, 'r') as fin:
            for line in tqdm(fin, total=n, desc='Calculating alignments'):
                dic = json.loads(line)
                question = dic['question']['stem']
                for choice in dic['question']['choices']:
                    answer = choice['text']
                    adj, concepts, _, _ = adj_data.pop(0)
                    concepts = [id2concept[c].replace('_', ' ') for c in concepts]
                    offsets.append(offsets[-1] + len(concepts))
                    for concept in concepts:
                        input_ids, input_mask, segment_ids, span = convert_qa_concept_to_bert_input(tokenizer, question, answer, concept, max_seq_length)
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_segment_ids.append(segment_ids)
                        all_span.append(span)

        assert len(adj_data) == 0
        check_path(cache_path)
        with open(cache_path, 'wb') as fout:
            pickle.dump((all_input_ids, all_input_mask, all_segment_ids, all_span, offsets), fout)
        print('Inputs dumped')

    all_input_ids, all_input_mask, all_segment_ids, all_span = [torch.tensor(x, dtype=torch.long) for x in [all_input_ids, all_input_mask, all_segment_ids, all_span]]
    all_span = all_span.to(device)

    concept_vecs = []
    n = all_input_ids.size(0)

    with torch.no_grad():
        for a in tqdm(range(0, n, batch_size), total=n // batch_size + 1, desc='Extracting features'):
            b = min(a + batch_size, n)
            batch = [x.to(device) for x in [all_input_ids[a:b], all_input_mask[a:b], all_segment_ids[a:b]]]
            outputs = model(*batch)
            hidden_states = outputs[-1][layer_id]
            mask = torch.arange(max_seq_length, device=device)[None, :]
            mask = (mask >= all_span[a:b, 0, None]) & (mask < all_span[a:b, 1, None])
            pooled = (hidden_states * mask.float().unsqueeze(-1)).sum(1)
            pooled = pooled / (all_span[a:b, 1].float() - all_span[a:b, 0].float() + 1e-5).unsqueeze(1)
            concept_vecs.append(pooled.cpu())
        concept_vecs = torch.cat(concept_vecs, 0).numpy()
        res = [concept_vecs[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print('done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--statement_path', default=None)
    parser.add_argument('--adj_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'], required=True)
    parser.add_argument('-ds', '--dataset', default='csqa', choices=['csqa', 'socialiqa', 'obqa'], required=True)
    parser.add_argument('--layer_id', type=int, default=-1)
    parser.add_argument('--max_seq_length', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cache_path', default=None)

    args = parser.parse_args()

    parser.set_defaults(statement_path=f'./data/{args.dataset}/statement/{args.split}.statement.jsonl',
                        adj_path=f'./data/{args.dataset}/graph/{args.split}.graph.adj.pk',
                        output_path=f'./data/{args.dataset}/concept_embs/{args.split}.bert-large-uncased.layer{args.layer_id}.pk',
                        cache_path=f'./data/{args.dataset}/concept_embs/{args.split}.inputs.pk')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    extract_bert_node_features_from_adj(cpnet_vocab_path=args.cpnet_vocab_path,
                                        statement_path=args.statement_path,
                                        adj_path=args.adj_path,
                                        output_path=args.output_path,
                                        max_seq_length=args.max_seq_length,
                                        device=device,
                                        batch_size=args.batch_size,
                                        layer_id=args.layer_id,
                                        cache_path=args.cache_path)
