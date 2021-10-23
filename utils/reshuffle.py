import argparse
import json
from tqdm import tqdm
import numpy as np

params = {
    'src':{
        'train': './data/{ds}/statement/train.statement.jsonl',
        'dev': './data/{ds}/statement/dev.statement.jsonl',
    },
    'tgt':{
        'train': './data/{ds}/statement/train.statement_.jsonl',
        'dev': './data/{ds}/statement/dev.statement_.jsonl',
    }
}



parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='obqa', choices=['csqa', 'obqa', 'socialiqa'])
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()
print(args)
print(args.ds)
np.random.seed(args.seed)

def read_file(filename):
    nrow = sum(1 for _ in open(filename, 'r'))
    li = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, total=nrow):
            json_line = json.loads(line)
            li.append(json_line)

    return li, len(li)

all = []
cnt = []
for split in ['train', 'dev']:
    li, length = read_file(params['src'][split].format(ds=args.ds))
    all.extend(li)
    cnt.append(length)
idxs = np.arange(len(all))
np.random.shuffle(idxs)

res = []
for length in cnt:
    res.append([all[idx] for idx in idxs[:length]])
    idxs = idxs[length:]


for split in ['train', 'dev']:
    with open(params['tgt'][split].format(ds=args.ds), 'w', encoding='utf-8') as fout:
        for item in tqdm(res[0], total=len(res[0])):
            fout.write(json.dumps(item) + '\n')
        res.pop(0)






