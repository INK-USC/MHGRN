import torch
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from transformers import (BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

try:
    from .conceptnet import merged_relations
except ModuleNotFoundError:
    from conceptnet import merged_relations
try:
    from .utils import check_path
except:
    from utils import check_path
import json

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetModel, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

concept2id = None
id2concept = None
relation2id = None
id2relation = None
templates = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def generate_triples_from_adj(adj_pk_path, mentioned_cpt_path, cpnet_vocab_path, triple_path):
    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    with open(mentioned_cpt_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    mentioned_concepts = [([concept2id[ac] for ac in item["ac"]] + [concept2id[qc] for qc in item["qc"]]) for item in data]

    with open(adj_pk_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)

    n_samples = len(adj_concept_pairs)
    triples = []
    mc_triple_num = []
    for idx, (adj_data, mc) in tqdm(enumerate(zip(adj_concept_pairs, mentioned_concepts)),
                                    total=n_samples, desc='loading adj matrices'):
        adj, concepts, _, _ = adj_data
        mapping = {i: (concepts[i]) for i in range(len(concepts))}  # index to corresponding grounded concept id
        ij = adj.row
        k = adj.col
        n_node = adj.shape[1]
        n_rel = 2 * adj.shape[0] // n_node
        i, j = ij // n_node, ij % n_node

        j = np.array([mapping[j[idx]] for idx in range(len(j))])
        k = np.array([mapping[k[idx]] for idx in range(len(k))])

        mc2mc_mask = np.isin(j, mc) & np.isin(k, mc)
        mc2nmc_mask = np.isin(j, mc) | np.isin(k, mc)
        others_mask = np.invert(mc2nmc_mask)
        mc2nmc_mask = ~mc2mc_mask & mc2nmc_mask
        mc2mc = i[mc2mc_mask], j[mc2mc_mask], k[mc2mc_mask]
        mc2nmc = i[mc2nmc_mask], j[mc2nmc_mask], k[mc2nmc_mask]
        others = i[others_mask], j[others_mask], k[others_mask]
        [i, j, k] = [np.concatenate((a, b, c), axis=-1) for (a, b, c) in zip(mc2mc, mc2nmc, others)]
        triples.append((i, j, k))
        mc_triple_num.append(len(mc2mc) + len(mc2nmc))

        # i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
        # mask = np.isin(j, mc)
        # inverted_mask = np.invert(mask)
        # masked = i[mask], j[mask], k[mask]
        # mc_triple_num.append(len(masked[0]))
        # remaining = i[inverted_mask], j[inverted_mask], k[inverted_mask]
        # [i, j, k] = [np.concatenate((m, r), axis=-1) for (m, r) in zip(masked, remaining)]
        # triples.append((i, j, k))  # i: relation, j: head, k: tail

    check_path(triple_path)
    with open(triple_path, 'wb') as fout:
        pickle.dump((triples, mc_triple_num), fout)
        print(f"Triples saved to {triple_path}")


def load_templates(str_template_path):
    global templates
    templates = {}
    with open(str_template_path, encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("["):
                rel = line.split('/')[0][1:]
                if rel.endswith(']'):
                    rel = rel[:-1]
                template[rel] = []
            elif "#SUBJ#" in line and "#OBJ#" in line:
                templates[rel].append(line)


def generate_triple_string_per_inst(triples):
    global id2concept, id2relation
    res = []
    for h, r, t in zip(*triples):
        if r > 17:  # magic number
            continue
        relation = id2relation[r]
        template = templates[relation]
        head, tail = [id2concept[x].replace('_', ' ').split() for x in [h, t]]
        head_len, tail_len = [len(x) for x in [head, tail]]
        template_list = template.split()
        subj_idx, obj_idx = [template_list.index(x) for x in ["#SUBJ", "#OBJ"]]
        subj_start = subj_idx if subj_idx < obj_idx else (subj_idx + tail_len - 1)
        subj_end = subj_idx + head_len
        obj_start = obj_idx if obj_idx < subj_idx else (obj_idx + head_len - 1)
        obj_end = obj_start + tail_len
        string = template.replace("#SUBJ#", ' '.join(head)).replace("#OBJ#", ''.join(tail))
        res.append({'str': string, 'subj_start': subj_start, 'subj_end': subj_end, 'obj_start': obj_start, 'obj_end': obj_end, 'subj_id': h, 'obj_id': t})

    return res


def generate_triple_string(str_template_path, cpnet_vocab_path, triple_path, output_path, num_processes=1):
    global concept2id, id2concept, relation2id, id2relation, template

    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if template is None:
        load_templates(str_template_path=str_template_path)
    with open(triple_path, 'rb') as fin:
        triples, _ = pickle.load(fin)
    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for res in tqdm(p.imap(generate_triple_string_per_inst, triples), total=len(triples)):
            fout.write(json.dumps(res) + '\n')


if __name__ == "__main__":
    pass
# generate_triples_from_adj(adj_pk_path="../data/csqa/graph/test.graph.adj.pk",
#                           mentioned_cpt_path="../data/csqa/grounded/test.grounded.jsonl",
#                           cpnet_vocab_path="../data/cpnet/concept.txt",
#                           triple_path="../test_triple")
