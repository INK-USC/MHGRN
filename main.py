import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import copy
import time
import argparse
from modeling.modeling_kagnet import *
from utils.utils import *


def evaluate_accuracy(eval_set, model, model_type):
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for statements, correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts, qa_path_num, path_len in eval_set:
            batch_size, num_choice = statements.size(0), statements.size(1)
            flat_statements = statements.view(batch_size * num_choice, -1)
            flat_qa_pairs = sum(qa_pairs, [])
            flat_cpt_paths = sum(cpt_paths, [])
            flat_rel_paths = sum(rel_paths, [])
            flat_qa_path_num = sum(qa_path_num, [])
            flat_path_len = sum(path_len, [])
            if model_type == 'kagnet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, graphs,
                                    concept_mapping_dicts, flat_qa_path_num, flat_path_len)
            elif model_type == 'kernet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, flat_qa_path_num, flat_path_len)
            elif model_type == 'relation_net':
                flat_logits = model(flat_statements, flat_qa_pairs)
            elif model_type == 'gcn':
                flat_logits = model(flat_statements, graphs)
            elif model_type == 'bert':
                flat_logits = model(flat_statements)
            flat_logits = flat_logits.view(-1, num_choice)
            _, pred = flat_logits.max(1)
            n_correct += (pred == correct_labels).sum().item()
    model.train()
    return n_correct / len(eval_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='kagnet', choices=['kernet', 'kagnet', 'relation_net', 'gcn', 'bert'], help='model type')
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--glove', default='./data/glove/glove.6B.300d.txt', help='path to GloVe embeddings')
    parser.add_argument('--ent_emb', default='./data/transe/glove.transe.sgd.ent.npy', help='path to TransE entity embeddings')
    parser.add_argument('--rel_emb', default='./data/transe/glove.transe.sgd.rel.npy', help='path to TransE relation embeddings')

    # datasets
    parser.add_argument('--train_statements', default='./data/csqa/statement/train.statement.jsonl')
    parser.add_argument('--train_graphs', default='./data/csqa/graph/train.graph.jsonl')
    parser.add_argument('--train_paths', default='./data/csqa/paths/train.paths.pruned.jsonl')
    parser.add_argument('--train_feats', default='./data/csqa/bert/train.bert.large.layer-2.epoch1.npy')
    parser.add_argument('--dev_statements', default='./data/csqa/statement/dev.statement.jsonl')
    parser.add_argument('--dev_graphs', default='./data/csqa/graph/dev.graph.jsonl')
    parser.add_argument('--dev_paths', default='./data/csqa/paths/dev.paths.pruned.jsonl')
    parser.add_argument('--dev_feats', default='./data/csqa/bert/dev.bert.large.layer-2.epoch1.npy')
    parser.add_argument('--test_statements', default='./data/csqa/statement/test.statement.jsonl')
    parser.add_argument('--test_graphs', default='./data/csqa/graph/test.graph.jsonl')
    parser.add_argument('--test_paths', default='./data/csqa/paths/test.pruned.jsonl')
    parser.add_argument('--test_feats', default='./data/csqa/bert/test.finetuned.large.-2.npy')
    parser.add_argument('--sample_train', default=None, type=int, help='sample N examples from the training set')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')
    parser.add_argument('--use_roberta', default=False, type=bool_flag, nargs='?', const=True, help='use RoBERTa features')

    # model architecture
    parser.add_argument('--num_choice', default=5, type=int, help='number choices per qeustion')
    parser.add_argument('--lstm_dim', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--lstm_layer_num', default=1, type=int, help='number of LSTM layers')
    parser.add_argument('--bidirect', default=False, type=bool_flag, nargs='?', const=True, help='use bidirectional LSTM')
    parser.add_argument('--sent_dim', default=1024, type=int, help='dimensionality of the statement vector')
    parser.add_argument('--qas_encoded_dim', default=128, type=int, help='dimensionality of an encoded (qc, ac) pair')
    parser.add_argument('--num_random_paths', default=None, type=int, help='number random paths to sample during training')
    parser.add_argument('--graph_hidden_dim', default=50, type=int, help='number of hidden units of the GCN')
    parser.add_argument('--graph_output_dim', default=25, type=int, help='number of output units of the GCN')
    parser.add_argument('--fc_hidden_dim', default=64, type=int, help='number of hidden units of the FC layers')
    parser.add_argument('--freeze_emb', default=False, type=bool_flag, nargs='?', const=True, help='freeze embedding layer')

    # regularization
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout probability')
    parser.add_argument('--wdecay', default=1e-5, type=float, help='l2 weight decay strength')

    # other model options
    parser.add_argument('--path_cutoff', default=4, type=int)

    # optimization
    parser.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--eval_batch_size', default=200, type=int)
    parser.add_argument('--max_steps', default=50000, type=int)
    parser.add_argument('--max_steps_before_stop', default=500, type=int)

    # logging
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--save_dir', default='./saved_models/kagnet/', help='model output directory')
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')

    args = parser.parse_args()

    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)
    if args.model == 'gcn':
        parser.set_defaults(eval_batch_size=2000)
    if args.path_cutoff >= 4:
        parser.set_defaults(eval_batch_size=50)
    if args.use_roberta:
        parser.set_defaults(train_feats='data/csqa/bert/train.roberta.cls.layer-2.npy')
        parser.set_defaults(dev_feats='data/csqa/bert/dev.roberta.cls.layer-2.npy')
        parser.set_defaults(test_feats='data/csqa/bert/test.roberta.cls.layer-2.npy')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_acc,dev_acc\n')

    cp_emb, rel_emb = np.load(args.ent_emb), np.load(args.rel_emb)
    concept_num, concept_dim = cp_emb.shape[0] + 1, cp_emb.shape[1]  # add a dummy concept
    cp_emb = torch.tensor(np.insert(cp_emb, 0, np.zeros((1, concept_dim)), 0))
    relation_num, relation_dim = rel_emb.shape[0] * 2 + 1, rel_emb.shape[1]  # for inverse and dummy relations
    rel_emb = np.concatenate((rel_emb, rel_emb), 0)
    rel_emb = torch.tensor(np.insert(rel_emb, 0, np.zeros((1, relation_dim)), 0))

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.model == "kagnet":
        model = KnowledgeAwareGraphNetwork(args.sent_dim, concept_dim, relation_dim, concept_num, relation_num,
                                           args.qas_encoded_dim, cp_emb, rel_emb,
                                           args.lstm_dim, args.lstm_layer_num, device, args.graph_hidden_dim, args.graph_output_dim,
                                           dropout=args.dropout, bidirect=args.bidirect, num_random_paths=args.num_random_paths,
                                           path_attention=True, qa_attention=True)
    elif args.model == 'kernet':
        model = KnowledgeEnhancedRelationNetwork(args.sent_dim, concept_dim, relation_dim,
                                                 concept_num, relation_num, args.qas_encoded_dim,
                                                 cp_emb, rel_emb,
                                                 args.lstm_dim, args.lstm_layer_num, device,
                                                 dropout=args.dropout, bidirect=args.bidirect,
                                                 num_random_paths=args.num_random_paths,
                                                 path_attention=True, qa_attention=True)

    elif args.model == "relation_net":
        model = RelationNetwork(concept_dim, concept_num, cp_emb,
                                args.sent_dim, args.qas_encoded_dim, device=device)
    elif args.model == 'gcn':
        model = GCNSent(args.sent_dim, args.fc_hidden_dim, concept_dim, args.graph_hidden_dim, args.graph_output_dim,
                        cp_emb, dropout=args.dropout)
    elif args.model == "bert":
        model = KagNetMLP(args.sent_dim, args.fc_hidden_dim, 1, args.dropout)
    if args.freeze_emb:
        freeze_net(model.concept_emd)
        # if hasattr(model, 'relation_emd'):
        #     freeze_net(model.relation_emd)
    model.to(device)

    train_set = CSQADataLoader(args.train_statements, args.train_paths, args.train_graphs, args.train_feats,
                               args.batch_size, device, num_choice=args.num_choice, cut_off=args.path_cutoff,
                               end=args.sample_train, use_cache=args.use_cache)
    train_set_copy = CSQADataLoader(args.train_statements, args.train_paths, args.train_graphs, args.train_feats,
                                    args.eval_batch_size,       device, num_choice=args.num_choice, cut_off=args.path_cutoff,
                                    end=args.sample_train, use_cache=args.use_cache)  # for evaluating train accuracy
    dev_set = CSQADataLoader(args.dev_statements, args.dev_paths, args.dev_graphs, args.dev_feats, args.eval_batch_size,
                             device, num_choice=args.num_choice, cut_off=args.path_cutoff, use_cache=args.use_cache)

    print('len(train_set): {}   len(dev_set): {}'.format(len(train_set), len(dev_set)))
    print()

    print('parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, amsgrad=True)
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    print('-' * 52)
    global_step = 0
    last_best_step = 0
    best_dev_acc = 0
    total_loss = 0
    end_flag = False
    start_time = time.time()
    model.train()
    while not end_flag:
        for statements, correct_labels, graphs, cpt_paths, rel_paths, qa_pairs, concept_mapping_dicts, qa_path_num, path_len in train_set:
            optimizer.zero_grad()

            batch_size = statements.size(0)
            flat_statements = statements.view(batch_size * args.num_choice, -1)
            # n * num_qa_pair x 2
            flat_qa_pairs = sum(qa_pairs, [])
            # n * num_path x max_path_len
            flat_cpt_paths = sum(cpt_paths, [])
            flat_rel_paths = sum(rel_paths, [])
            flat_qa_path_num = sum(qa_path_num, [])
            flat_path_len = sum(path_len, [])

            if args.model == 'kagnet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths,
                                    graphs, concept_mapping_dicts, flat_qa_path_num, flat_path_len)
            elif args.model == 'kernet':
                flat_logits = model(flat_statements, flat_qa_pairs, flat_cpt_paths, flat_rel_paths, flat_qa_path_num, flat_path_len)
            elif args.model == 'relation_net':
                flat_logits = model(flat_statements, flat_qa_pairs)
            elif args.model == 'gcn':
                flat_logits = model(flat_statements, graphs)
            elif args.model == "bert":
                flat_logits = model(flat_statements)
            flat_logits = flat_logits.view(-1)

            if args.loss == 'margin_rank':
                flat_logits = F.sigmoid(flat_logits)
                correct_mask = F.one_hot(correct_labels, num_classes=args.num_choice).view(-1)  # of length batch_size*num_choice
                correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, args.num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                y = wrong_logits.new_ones((wrong_logits.size(0),))
                loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
            elif args.loss == 'cross_entropy':
                loss = loss_func(flat_logits.view(-1, args.num_choice), correct_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, total_loss, ms_per_batch))
                total_loss = 0
                start_time = time.time()

            if (global_step + 1) % args.eval_interval == 0:
                model.eval()
                train_acc = evaluate_accuracy(train_set_copy, model, args.model)
                dev_acc = evaluate_accuracy(dev_set, model, args.model)
                print('-' * 52)
                print('| step {:5} | train_acc {:7.4f} | dev_acc {:7.4f} |'.format(global_step, train_acc, dev_acc))
                print('-' * 52)
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, train_acc, dev_acc))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    last_best_step = global_step
                    torch.save(model, model_path)
                    print(f'model saved to {model_path}')
                model.train()
                start_time = time.time()

            global_step += 1
            if global_step - last_best_step >= args.max_steps_before_stop:
                end_flag = True
                break

        train_set.reshuffle()

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f}'.format(best_dev_acc))


def eval(args):
    raise NotImplementedError()  # TODO


if __name__ == '__main__':
    main()
