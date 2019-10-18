import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

try:
    from .utils import check_path
except ModuleNotFoundError:
    from utils import check_path

colors = ['b', 'r', 'g', 'k', 'y', 'c']

categories = {
    'animals': ['dog', 'cat', 'horse', 'bird', 'mouse', 'dogs', 'cats', 'horses', 'birds', 'mice', 'fox',
                'rat', 'foxes', 'rats', 'lizard', 'lizards', 'cat', 'cats', 'cattle', 'goat', 'goats', 'pig',
                'pigs', 'chicken', 'chickens', 'duck', 'ducks', 'goose', 'geese', 'pigeon', 'pigeons',
                'turkey', 'turkeys', 'buffalo', 'sheep', 'ant', 'ants', 'bee', 'bees', 'bear', 'bears',
                'camel', 'camels', 'fish', 'cobra', 'cobras', 'deer', 'dolphin', 'dolphins', 'eagle',
                'eagles', 'falcon', 'falcons', 'giraffe', 'giraffes', 'hawk', 'hawks', 'jaguar', 'jaguars',
                'lion', 'lions', 'leopard', 'leopards', 'mammal', 'shark', 'sharks', 'octopus', 'otter',
                'otters', 'rabbit', 'rabbits', 'squirrel', 'squirrels', 'salmon', 'termite', 'termites',
                'tiger', 'tigers', 'wolf', 'wolves', 'worm', 'worms'],

    'vehicles': ['car', 'bus', 'motorcycle', 'airplane', 'boat', 'cars', 'buses', 'motorcycles', 'airplanes',
                 'boats', 'bicycle', 'bicycles', 'helicopter', 'helicopters', 'bike', 'bikes', 'ship', 'ships',
                 'taxi', 'taxis', 'train', 'trains', 'van', 'vans', 'auto', 'autos', 'highway', 'highways',
                 'street', 'streets', 'lane', 'lanes', 'transport', 'transports', 'ferry', 'ferries', 'barge',
                 'barges', 'sailboat', 'sailboats', 'steamboat', 'steamboats', 'motorcar', 'motorcars',
                 'railroad', 'railroads', 'railway', 'railways', 'motorway', 'motorways', 'road', 'roads'],

    'grammar': ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
                'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
                'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
                'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'],

    'verb': ['walk', 'jump', 'run', 'leap', 'hop', 'skip', 'sprint', 'crawl', 'stumble', 'hurry', 'rush', 'trot',
             'hike', 'stroll', 'jog', 'flee', 'walks', 'jumps', 'runs', 'leaps', 'hops', 'skips', 'sprints',
             'crawls', 'stumbles', 'hurries', 'rushes', 'trots', 'hikes', 'strolls', 'jogs', 'flees', 'walked',
             'jumped', 'ran', 'leapt', 'hopped', 'skipped', 'sprinted', 'crawled', 'stumbled', 'hurried',
             'rushed', 'trotted', 'hiked', 'strolled', 'jogged', 'fled', 'walking', 'jumping', 'running',
             'leaping', 'hopping', 'skipping', 'sprinting', 'crawling', 'stumbling', 'hurrying', 'rushing',
             'trotting', 'hiking', 'strolling', 'jogging', 'fleeing'],
}


def plot_tsne(embeddings, vocab, output_path, format='png', random_state=0, normalize=False):
    """
    embeddings: numpy.array of shape (num_concepts, emb_dim)
    vocab: list[str]
    output_path: str
    format: str (optional, default 'png')
    random_state: str (optional, default 0)
    """
    assert len(embeddings) == len(vocab)
    concept2id = {w: i for i, w in enumerate(vocab)}
    all_ids = []
    offsets = [0]
    for cate in ['animals', 'vehicles', 'grammar', 'verb']:
        cate_ids = [concept2id[s] for s in categories[cate] if s in concept2id]
        all_ids += cate_ids
        offsets.append(offsets[-1] + len(cate_ids))
        oov_rate = 1 - len(cate_ids) / len(categories[cate])
        print('[tsne] {} oov rate = {}'.format(cate, oov_rate))

    original_x = embeddings[np.array(all_ids)]
    if normalize:
        original_x = original_x / np.sqrt((original_x ** 2).sum(1))[:, np.newaxis]
    reduced_x = TSNE(2, random_state=random_state).fit_transform(original_x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, cate in enumerate(['animals', 'vehicles', 'grammar', 'verb']):
        cate_x = reduced_x[offsets[i]:offsets[i + 1]]
        ax.scatter(cate_x[:, 0], cate_x[:, 1], label=cate, color=colors[i])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.legend(fancybox=True, shadow=True)
    check_path(output_path)
    fig.savefig(output_path, format=format, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='./images/transe.png')
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--ent_emb', default='./data/transe/glove.transe.sgd.ent.npy')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    with open(args.cpnet_vocab_path, 'r', encoding='utf-8') as fin:
        id2concept = [w.strip() for w in fin]

    ent_emb = np.load(args.ent_emb)

    plot_tsne(ent_emb, id2concept, args.output, 'png', args.seed)


if __name__ == '__main__':
    main()
