import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
from utils.conceptnet import merged_relations
from utils.layers import *
from utils.utils import *

id2concept = None
concept2id = None
id2relation = None
relation2id = None
cpnet = None
cpnet_simple = None


def get_rel_paths(path):
    if len(path) == 2:
        rel_list = cpnet[path[0]][path[1]]
        seen = set()
        res = [r['rel'] for r in rel_list.values() if r['rel'] not in seen and (seen.add(r['rel']) or True)]
        res = [(r,) for r in res]
        return res
    elif len(path) == 3:
        rel_list1 = cpnet[path[0]][path[1]]
        seen1 = set()
        res1 = [r['rel'] for r in rel_list1.values() if r['rel'] not in seen1 and (seen1.add(r['rel']) or True)]
        rel_list2 = cpnet[path[1]][path[2]]
        seen2 = set()
        res2 = [r['rel'] for r in rel_list2.values() if r['rel'] not in seen2 and (seen2.add(r['rel']) or True)]
        res = [(r1, r2) for r1 in res1 for r2 in res2]
        return res
    else:
        raise ValueError('Invalid path length')


def find_paths_qa_concept_pair(source: str, target: str, ifprint=False):
    """
    find paths for a (question concept, answer concept) pair
    source and target is text
    """
    global cpnet, cpnet_simple, concept2id, id2concept, relation2id, id2relation

    s = concept2id[source]
    t = concept2id[target]

    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return []

    all_path = []
    for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=2):
        if len(p) >= 2:  # skip paths of length 1
            all_path.append(p)

    res = []
    seen = set()
    for p in all_path:
        for rpath in get_rel_paths(p):
            if rpath not in seen:
                res.append({"qc": s, "ac": t, "rel": rpath})
                seen.add(rpath)
    return res


def find_relational_paths_qa_pair(qa_pair):
    acs, qcs = qa_pair
    pfr_qa = {'acs': acs, 'qcs': qcs}
    rel_list = []
    for ac in acs:
        for qc in qcs:
            rel_list += find_paths_qa_concept_pair(qc, ac)
    pfr_qa['paths'] = rel_list
    return pfr_qa


def find_relational_paths(cpnet_vocab_path, cpnet_graph_path, grounded_path, output_path, num_processes, use_cache):
    if use_cache and os.path.exists(output_path):
        print(f'using cached relational paths from {output_path}')
        return

    def get_cpnet_simple(nx_graph):
        cpnet_simple = nx.Graph()
        for u, v, data in nx_graph.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        return cpnet_simple

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        with open(cpnet_vocab_path, 'r', encoding='utf-8') as fin:
            id2concept = [w.strip() for w in fin]
        concept2id = {w: i for i, w in enumerate(id2concept)}
        id2relation = merged_relations.copy()
        id2relation += ['*' + r for r in id2relation]
        relation2id = {r: i for i, r in enumerate(id2relation)}
    if cpnet is None or cpnet_simple is None:
        cpnet = nx.read_gpickle(cpnet_graph_path)
        cpnet_simple = get_cpnet_simple(cpnet)

    with open(grounded_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    data = [[item["ac"], item["qc"]] for item in data]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for pfr_qa in tqdm(p.imap(find_relational_paths_qa_pair, data), total=len(data), desc='Finding relational paths'):
            fout.write(json.dumps(pfr_qa) + '\n')

    print(f'paths saved to {output_path}')
    print()


def find_relational_paths(cpnet_vocab_path, cpnet_graph_path, grounded_path, output_path, num_processes, use_cache):
    if use_cache and os.path.exists(output_path):
        print(f'using cached relational paths from {output_path}')
        return

    def get_cpnet_simple(nx_graph):
        cpnet_simple = nx.Graph()
        for u, v, data in nx_graph.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        return cpnet_simple

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        with open(cpnet_vocab_path, 'r', encoding='utf-8') as fin:
            id2concept = [w.strip() for w in fin]
        concept2id = {w: i for i, w in enumerate(id2concept)}
        id2relation = merged_relations.copy()
        id2relation += ['*' + r for r in id2relation]
        relation2id = {r: i for i, r in enumerate(id2relation)}
    if cpnet is None or cpnet_simple is None:
        cpnet = nx.read_gpickle(cpnet_graph_path)
        cpnet_simple = get_cpnet_simple(cpnet)

    with open(grounded_path, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]
    data = [[item["ac"], item["qc"]] for item in data]

    with Pool(num_processes) as p, open(output_path, 'w', encoding='utf-8') as fout:
        for pfr_qa in tqdm(p.imap(find_relational_paths_qa_pair, data), total=len(data), desc='Finding relational paths'):
            fout.write(json.dumps(pfr_qa) + '\n')

    print(f'paths saved to {output_path}')
    print()
