import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import (AdamW, ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
import random
import argparse
from modeling.modeling_rgcn import *
from utils.relpath_utils import *
from utils.datasets import *
from utils.embedding import *


def cal_2hop_rel_emb(rel_emb):
    n_rel = rel_emb.shape[0]
    u, v = np.meshgrid(np.arange(n_rel), np.arange(n_rel))
    expanded = rel_emb[v.reshape(-1)] + rel_emb[u.reshape(-1)]
    return np.concatenate([rel_emb, expanded], 0)


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='bert-large-uncased', help='encoder type')
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('-ih', '--inhouse', default=True, type=bool_flag, nargs='?', const=True, help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default='./data/{dataset}/inhouse_split_qids.txt', help='qids of the in-house training set')

    parser.add_argument('--glove_npy', default='./data/glove/glove.6B.300d.npy', help='path to GloVe npy file')
    parser.add_argument('--glove_vocab', default='./data/glove/glove.vocab', help='path to GloVe vocab file')

    # for finding relation paths
    parser.add_argument('--emb_sources', default=['transe'], choices=['transe', 'numberbatch'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--ent_emb', default=['./data/transe/glove.transe.sgd.ent.npy', './data/transe/concept.nb.npy'], help='path to TransE entity embeddings')
    parser.add_argument('--rel_emb', default='./data/transe/glove.transe.sgd.rel.npy', help='path to TransE relation embeddings')
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cpnet_graph_path', default='./data/cpnet/conceptnet.en.pruned.graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    # datasets
    parser.add_argument('--dataset', default='csqa', choices=['csqa', 'socialiqa', 'obqa'], help='dataset')
    parser.add_argument('--num_choice', default=5, type=int, help='number choices per question')
    parser.add_argument('--num_relation', default=35, type=int, help='number of relations')
    parser.add_argument('--train_statements', default='./data/{dataset}/statement/train.statement.jsonl')
    parser.add_argument('--train_adj', default='./data/{dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--train_tokenized', default='./data/{dataset}/tokenized/train.tokenized.txt')
    parser.add_argument('--dev_statements', default='./data/{dataset}/statement/dev.statement.jsonl')
    parser.add_argument('--dev_adj', default='./data/{dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--dev_tokenized', default='./data/{dataset}/tokenized/dev.tokenized.txt')
    parser.add_argument('--test_statements', default='./data/{dataset}/statement/test.statement.jsonl')
    parser.add_argument('--test_adj', default='./data/{dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--test_tokenized', default='./data/{dataset}/tokenized/test.tokenized.txt')
    parser.add_argument('--layer_id', default=-1, type=int, help='encoder layer ID to use as features')
    parser.add_argument('--sample_train', default=None, type=int, help='sample N examples from the training set')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--max_node_num', default=200, type=int)

    # model architecture
    parser.add_argument('--ablation', default=None, choices=['no_node_type_emb'], help='run ablation test')
    parser.add_argument('--diag_decompose', default=False, type=bool_flag, nargs='?', const=True, help='use diagonal decomposition')
    parser.add_argument('-k', '--node_mask_size', default=0, type=int, help='number of selected nodes at each timestep (0 to disable')
    parser.add_argument('--num_basis', default=8, type=int, help='number of basis (0 to disable basis decomposition)')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--freeze_lstm_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze lstm input embedding layer')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_layer_num', default=1, type=int, help='number of GNN layers')
    parser.add_argument('--dk', default=50, type=int, help='d_k')
    parser.add_argument('--dv', default=50, type=int, help='d_v')
    parser.add_argument('--fc_dim', default=200, type=int, help='hidden dim of the fully-connected layers')
    parser.add_argument('--fc_layer_num', default=1, type=int, help='number of the fully-connected layers')
    parser.add_argument('--node_type_emb_dim', default=20, type=int, help='dimensionality of node type embedding')

    parser.add_argument('--encoder_hidden_size', default=128, type=int, help='number of LSTM hidden units')
    parser.add_argument('--encoder_num_layers', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--encoder_bidirectional', default=True, type=bool_flag, nargs='?', const=True, help='use bidirectional LSTM')

    # regularization
    parser.add_argument('--e_dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--e_dropouti', type=float, default=0.4, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--e_dropoutr', type=float, default=0.3, help='dropout for rnn hidden units (0 = no dropout)')
    parser.add_argument('--e_dropouto', type=float, default=0.3, help='dropout for rnn outputs (0 = no dropout')
    parser.add_argument('--e_dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.3, help='dropout for fully-connected layers')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='l2 weight decay strength')

    # optimization
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'cross_entropy'], help='model type')
    parser.add_argument('--optim', default='adamw', choices=['adamw', 'adam'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='warmup_constant', choices=['fixed', 'warmup_linear', 'warmup_constant'], help='learning rate scheduler')
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('-mbs', '--mini_batch_size', default=4, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--n_epochs', default=8, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=5, type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--max_steps', default=8000, type=int)
    parser.add_argument('--max_steps_before_stop', default=3000, type=int)

    # others
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--eval_train', default=False, type=bool_flag, nargs='?', const=True, help='evaluate training set')
    parser.add_argument('--eval_interval', default=160, type=int)
    parser.add_argument('--save_dir', default='./saved_models/{dataset}/rgcn/', help='model output directory')
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')

    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)

    # set dataset defaults
    parser.set_defaults(num_choice=dataset_num_choice[args.dataset],
                        inhouse=(dataset_setting[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset),
                        save_dir=args.save_dir.format(dataset=args.dataset))
    parser.set_defaults(test_statements=None, test_adj=None, test_tokenized=None)
    for split in ('train', 'dev') if args.dataset in dataset_no_test else ('train', 'dev', 'test'):
        for attribute in ('statements', 'adj', 'tokenized'):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    dic = {'transe': 0, 'numberbatch': 1}
    cp_emb = np.concatenate([np.load(args.ent_emb[dic[source]]) for source in args.emb_sources], 1)
    cp_emb = torch.tensor(cp_emb)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        dataset = LMRGCNDataLoader(train_statement_path=args.train_statements, train_adj_path=args.train_adj,
                                   dev_statement_path=args.dev_statements, dev_adj_path=args.dev_adj,
                                   test_statement_path=args.test_statements, test_adj_path=args.test_adj,
                                   batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                   model_name=args.encoder, max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                   is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids, train_tokenized_path=args.train_tokenized,
                                   dev_tokenized_path=args.dev_tokenized, test_tokenized_path=args.test_tokenized, freq_cutoff=3)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        lstm_dic = {}
        if args.encoder in ('lstm',):
            emb_x = load_vectors_from_npy_with_vocab(args.glove_npy, args.glove_vocab, dataset.vocab)
            emb_x = torch.tensor(emb_x)

            lstm_dic = {'vocab_size': emb_x.size(0), 'emb_size': emb_x.size(1), 'hidden_size': args.encoder_hidden_size, 'num_layers': args.encoder_num_layers,
                        'bidirectional': args.encoder_bidirectional, 'pretrained_emb': emb_x,
                        'emb_p': args.e_dropoute, 'input_p': args.e_dropouti, 'output_p': args.e_dropouto, 'hidden_p': args.e_dropoutr, }

        model = LMRGCN(args.encoder, lstm_dic, num_concepts=concept_num, num_relations=args.num_relation, num_basis=args.num_basis,
                       concept_dim=concept_dim, d_k=args.dk, d_v=args.dv, num_gnn_layers=args.gnn_layer_num,
                       num_attention_heads=args.att_head_num, fc_dim=args.fc_dim, num_fc_layers=args.fc_layer_num,
                       p_gnn=args.dropoutg, p_fc=args.dropoutf,
                       pretrained_concept_emb=cp_emb, diag_decompose=args.diag_decompose, ablation=args.ablation)
        if args.freeze_ent_emb:
            freeze_net(model.decoder.concept_emb)
        if args.freeze_lstm_emb and args.encoder in ('lstm',):
            freeze_net(model.encoder.emb.emb)
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(grouped_parameters, amsgrad=True)
    else:
        optimizer = AdamW(grouped_parameters)
    if args.encoder in ('lstm',):
        scheduler = None
    else:
        if args.lr_schedule == 'fixed':
            scheduler = ConstantLRSchedule(optimizer)
        elif args.lr_schedule == 'warmup_constant':
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        elif args.lr_schedule == 'warmup_linear':
            max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=(max_steps * args.warmup_proportion), t_total=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print('-' * 71)
    global_step, last_best_step = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits, _ = model(*[x[a:b] for x in input_data])

                    if args.loss == 'margin_rank':
                        flat_logits = logits.view(-1)
                        correct_mask = F.one_hot(labels, num_classes=args.num_choice).view(-1)  # of length batch_size*num_choice
                        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, args.num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                        wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                        y = wrong_logits.new_ones((wrong_logits.size(0),))
                        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                    elif args.loss == 'cross_entropy':

                        loss = loss_func(logits, labels[a:b])
                    loss = loss * (b - a) / bs
                    loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scheduler is not None:
                    scheduler.step()
                optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0] if scheduler is not None else 0.0, total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()

                if (global_step + 1) % args.eval_interval == 0:
                    model.eval()
                    dev_acc = evaluate_accuracy(dataset.dev(), model)
                    test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
                    if args.eval_train:
                        train_acc = evaluate_accuracy(dataset.train(), model)
                    print('-' * 71)
                    print('| step {:5} | train_acc {:7.4f} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(global_step, train_acc if args.eval_train else 0.0, dev_acc, test_acc))
                    print('-' * 71)
                    with open(log_path, 'a') as fout:
                        fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
                    if dev_acc >= best_dev_acc:
                        best_dev_acc = dev_acc
                        final_test_acc = test_acc
                        last_best_step = global_step
                        torch.save([model, args], model_path)
                        print(f'model saved to {model_path}')
                    model.train()
                    start_time = time.time()

                global_step += 1
                if args.encoder in ('lstm',) and (global_step >= args.max_steps or global_step - last_best_step >= args.max_steps_before_stop):
                    break
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f}'.format(best_dev_acc))
    print('final test acc: {:.4f}'.format(final_test_acc))


def eval(args):
    raise NotImplementedError()


def pred(args):
    raise NotImplementedError()


if __name__ == '__main__':
    main()
