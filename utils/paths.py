import numpy as np
from scipy import spatial
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import json
import random
import os
from .conceptnet import merged_relations
import pickle

__all__ = ['find_paths', 'score_paths', 'prune_paths']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_simple = None

concept_embs = None
relation_embs = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


##################### path finding #####################


def get_edge(src_concept, tgt_concept):
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]  # list of dicts
    seen = set()
    res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]  # get unique values from rel_list
    return res


def find_paths_qa_concept_pair(source: str, target: str, ifprint=False):
    """
    find paths for a (question concept, answer concept) pair
    source and target is text
    """
    global cpnet, cpnet_simple, concept2id, id2concept, relation2id, id2relation

    s = concept2id[source]
    t = concept2id[target]

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return

    # all_path = []
    # all_path_set = set()
    # for max_len in range(1, 5):
    #     for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
    #         path_str = "-".join([str(c) for c in p])
    #         if path_str not in all_path_set:
    #             all_path_set.add(path_str)
    #             all_path.append(p)
    #             print(len(p), path_str)
    #         if len(all_path) >= 100:  # top shortest 100 paths
    #             break
    #     if len(all_path) >= 100:  # top shortest 100 paths
    #         break
    # all_path.sort(key=len, reverse=False)

    all_path = []
    try:
        for p in nx.shortest_simple_paths(cpnet_simple, source=s, target=t):
            if len(p) > 5 or len(all_path) >= 100:  # top 100 paths
                break
            if len(p) >= 2:  # skip paths of length 1
                all_path.append(p)
    except nx.exception.NetworkXNoPath:
        pass

    pf_res = []
    for p in all_path:
        # print([id2concept[i] for i in p])
        rl = []
        for src in range(len(p) - 1):
            src_concept = p[src]
            tgt_concept = p[src + 1]

            rel_list = get_edge(src_concept, tgt_concept)
            rl.append(rel_list)
            if ifprint:
                rel_list_str = []
                for rel in rel_list:
                    if rel < len(id2relation):
                        rel_list_str.append(id2relation[rel])
                    else:
                        rel_list_str.append(id2relation[rel - len(id2relation)] + "*")
                print(id2concept[src_concept], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                if src + 1 == len(p) - 1:
                    print(id2concept[tgt_concept], end="")
        if ifprint:
            print()

        pf_res.append({"path": p, "rel": rl})
    return pf_res


def find_paths_from_adj_per_inst(input):
    adj, concepts, qm, am = input
    adj = adj.toarray()
    ij, k = adj.shape

    adj = np.any(adj.reshape(ij // k, k, k), axis=0)
    simple_schema_graph = nx.from_numpy_matrix(adj)
    mapping = {i: int(c) for (i, c) in enumerate(concepts)}
    simple_schema_graph = nx.relabel_nodes(simple_schema_graph, mapping)
    qcs, acs = concepts[qm].tolist(), concepts[am].tolist()
    pfr_qa = []
    lengths = []
    for ac in acs:
        for qc in qcs:
            if qc not in simple_schema_graph.nodes() or ac not in simple_schema_graph.nodes():
                print('QA pair doesn\'t exist in schema graph.')
                pf_res = None
                lengths.append([0] * 3)
            else:
                all_path = []
                try:
                    for p in nx.shortest_simple_paths(simple_schema_graph, source=qc, target=ac):
                        if len(p) >= 5:
                            break
                        if len(p) >= 2:  # skip paths of length 1
                            all_path.append(p)
                except nx.exception.NetworkXNoPath:
                    pass

                length = [len(x) for x in all_path]
                lengths.append([length.count(2), length.count(3), length.count(4)])
                pf_res = []
                for p in all_path:
                    rl = []
                    for src in range(len(p) - 1):
                        src_concept = p[src]
                        tgt_concept = p[src + 1]
                        rel_list = get_edge(src_concept, tgt_concept)
                        rl.append(rel_list)
                    pf_res.append({"path": p, "rel": rl})
            pfr_qa.append({"ac": ac, "qc": qc, "pf_res": pf_res})
    g = nx.convert_node_labels_to_integers(simple_schema_graph, label_attribute='cid')

    return pfr_qa, nx.node_link_data(g), lengths


def find_paths_qa_pair(qa_pair):
    acs, qcs = qa_pair
    pfr_qa = []
    for ac in acs:
        for qc in qcs:
            pf_res = find_paths_qa_concept_pair(qc, ac)
            pfr_qa.append({"ac": ac, "qc": qc, "pf_res": pf_res})
    return pfr_qa


##################### path scoring #####################


def score_triple(h, t, r, flag):
    res = -10
    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t
        # result  = (cosine_sim + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)
    return res


def score_triples(concept_id, relation_id, debug=False):
    global relation_embs, concept_embs, id2relation, id2concept
    concept = concept_embs[concept_id]
    relation = []
    flag = []
    for i in range(len(relation_id)):
        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)
        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        for j in range(len(relation_id[i])):
            if relation_id[i][j] >= 17:
                embs.append(relation_embs[relation_id[i][j] - 17])
                l_flag.append(1)
            else:
                embs.append(relation_embs[relation_id[i][j]])
                l_flag.append(0)
        relation.append(embs)
        flag.append(l_flag)

    res = 1
    for i in range(concept.shape[0] - 1):
        h = concept[i]
        t = concept[i + 1]
        score = score_triple(h, t, relation[i], flag[i])
        res *= score

    if debug:
        print("Num of concepts:")
        print(len(concept_id))
        to_print = ""
        for i in range(concept.shape[0] - 1):
            h = id2concept[concept_id[i]]
            to_print += h + "\t"
            for rel in relation_id[i]:
                if rel >= 17:
                    # 'r-' means reverse
                    to_print += ("r-" + id2relation[rel - 17] + "/  ")
                else:
                    to_print += id2relation[rel] + "/  "
        to_print += id2concept[concept_id[-1]]
        print(to_print)
        print("Likelihood: " + str(res) + "\n")

    return res


def score_qa_pairs(qa_pairs):
    statement_scores = []
    for qas in qa_pairs:
        statement_paths = qas["pf_res"]
        if statement_paths is not None:
            path_scores = []
            for path in statement_paths:
                assert len(path["path"]) > 1
                score = score_triples(concept_id=path["path"], relation_id=path["rel"])
                path_scores.append(score)
            statement_scores.append(path_scores)
        else:
            statement_scores.append(None)
    return statement_scores


def find_relational_paths_from_paths_per_inst(path_dic):
    qcs = set()
    acs = set()
    seen = set()
    rel_list = []
    for qa_pair_dic in path_dic:
        qcs.add(qa_pair_dic['qc'])
        acs.add(qa_pair_dic['ac'])
        if qa_pair_dic['pf_res'] is None:
            continue
        for path in qa_pair_dic['pf_res']:
            if len(path['path']) == 2:
                for r in path['rel'][0]:
                    if (path['path'][0], path['path'][-1], r) not in seen:
                        rel_list.append({'qc': path['path'][0], 'ac': path['path'][-1], 'rel': [r]})
                        seen.add((path['path'][0], path['path'][-1], r))
            elif len(path['path']) == 3:
                for r1 in path['rel'][0]:
                    for r2 in path['rel'][1]:
                        if (path['path'][0], path['path'][-1], r1, r2) not in seen:
                            rel_list.append({'qc': path['path'][0], 'ac': path['path'][-1], 'rel': [r1, r2]})
                            seen.add((path['path'][0], path['path'][-1], r1, r2))
    pfr_qa = {'acs': list(acs), 'qcs': list(qcs), 'paths': rel_list}
    return pfr_qa


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

def find_paths(grounded_path, cpnet_vocab_path, cpnet_graph_path, output_path, num_processes=1, random_state=0):
    print(f'generating paths for {grounded_path}...')
    random.seed(random_state)
    np.random.seed(random_state)

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    with open(grounded_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    data = [[item["ac"], item["qc"]] for item in data]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for pfr_qa in tqdm(p.imap(find_paths_qa_pair, data), total=len(data)):
            fout.write(json.dumps(pfr_qa) + '\n')

    print(f'paths saved to {output_path}')
    print()


def generate_path_and_graph_from_adj(adj_path, cpnet_graph_path, output_path, graph_output_path, num_processes=1, random_state=0, dump_len=False):
    print(f'generating paths for {adj_path}...')
    global cpnet
    if cpnet is None:
        cpnet = nx.read_gpickle(cpnet_graph_path)
    random.seed(random_state)
    np.random.seed(random_state)
    with open(adj_path, 'rb') as fin:
        adj_concept_pairs = pickle.load(fin)  # (adj, concepts, qm, am)
    all_len = []
    with Pool(num_processes) as p, open(output_path, 'w') as path_output, open(graph_output_path, 'w') as graph_output:
        for pfr_qa, graph, lengths in tqdm(p.imap(find_paths_from_adj_per_inst, adj_concept_pairs), total=len(adj_concept_pairs), desc='Searching for paths'):
            path_output.write(json.dumps(pfr_qa) + '\n')
            graph_output.write(json.dumps(graph) + '\n')
            all_len.append(lengths)
    if dump_len:
        with open(adj_path+'.len.pk', 'wb') as f:
            pickle.dump(all_len, f)
    print(f'paths saved to {output_path}')
    print(f'graphs saved to {graph_output_path}')
    print()



def score_paths(raw_paths_path, concept_emb_path, rel_emb_path, cpnet_vocab_path, output_path, num_processes=1, method='triple_cls'):
    print(f'scoring paths for {raw_paths_path}...')
    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global concept_embs, relation_embs
    if concept_embs is None:
        concept_embs = np.load(concept_emb_path)
    if relation_embs is None:
        relation_embs = np.load(rel_emb_path)

    if method != 'triple_cls':
        raise NotImplementedError()

    all_scores = []
    with open(raw_paths_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for statement_scores in tqdm(p.imap(score_qa_pairs, data), total=len(data)):
            fout.write(json.dumps(statement_scores) + '\n')

    print(f'path scores saved to {output_path}')
    print()


def prune_paths(raw_paths_path, path_scores_path, output_path, threshold, verbose=True):
    print(f'pruning paths for {raw_paths_path}...')
    ori_len = 0
    pruned_len = 0
    nrow = sum(1 for _ in open(raw_paths_path, 'r'))
    with open(raw_paths_path, 'r', encoding='utf-8') as fin_raw, \
            open(path_scores_path, 'r', encoding='utf-8') as fin_score, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line_raw, line_score in tqdm(zip(fin_raw, fin_score), total=nrow):
            qa_pairs = json.loads(line_raw)
            qa_pairs_scores = json.loads(line_score)
            for qas, qas_scores in zip(qa_pairs, qa_pairs_scores):
                ori_paths = qas['pf_res']
                if ori_paths is not None:
                    pruned_paths = [p for p, s in zip(ori_paths, qas_scores) if s >= threshold]
                    ori_len += len(ori_paths)
                    pruned_len += len(pruned_paths)
                    assert len(ori_paths) >= len(pruned_paths)
                    qas['pf_res'] = pruned_paths
            fout.write(json.dumps(qa_pairs) + '\n')

    if verbose:
        print("ori_len: {}   pruned_len: {}   keep_rate: {:.4f}".format(ori_len, pruned_len, pruned_len / ori_len))

    print(f'pruned paths saved to {output_path}')
    print()


def find_relational_paths_from_paths(pruned_paths_path, output_path, num_processes):
    print(f'extracting relational paths from {pruned_paths_path}...')
    with open(pruned_paths_path, 'r', encoding='utf-8') as fin:
        path_data = [json.loads(line) for line in fin]
    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for pfr_qa in tqdm(p.imap(find_relational_paths_from_paths_per_inst, path_data), total=len(path_data)):
            fout.write(json.dumps(pfr_qa) + '\n')
    print(f'paths saved to {output_path}')
    print()
