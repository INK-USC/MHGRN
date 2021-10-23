import random

from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)

from modeling.modeling_grn import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils.relpath_utils import *

DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
}


def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')


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
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred', 'decode'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/grn/', help='model output directory')

    # data
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--num_relation', default=34, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--train_embs', default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--dev_embs', default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--test_embs', default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')

    # model architecture
    parser.add_argument('-k', '--k', default=2, type=int, help='perform k-hop message passing at each layer')
    parser.add_argument('--ablation', default=[], choices=['no_trans', 'early_relu', 'no_att', 'ctx_trans', 'q2a_only',
                                                           'no_typed_transform', 'no_type_att', 'typed_pool', 'no_unary',
                                                           'detach_s_agg', 'detach_s_all', 'detach_s_pool', 'agg_self_loop',
                                                           'early_trans', 'pool_qc', 'pool_ac', 'pool_all',
                                                           'no_ent', 'no_rel', 'no_rel_att', 'no_1hop', 'fix_scale',
                                                           'no_lm'], nargs='*', help='run ablation test')
    parser.add_argument('-dd', '--diag_decompose', default=True, type=bool_flag, nargs='?', const=True, help='use diagonal decomposition')
    parser.add_argument('--num_basis', default=0, type=int, help='number of basis (0 to disable basis decomposition)')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--att_dim', default=50, type=int, help='dimensionality of the query vectors')
    parser.add_argument('--att_layer_num', default=1, type=int, help='number of hidden layers of the attention module')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--gnn_layer_num', default=1, type=int, help='number of GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--eps', type=float, default=1e-15, help='avoid numeric overflow')
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--init_rn', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--init_identity', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--fix_trans', default=False, type=bool_flag, nargs='?', const=True)

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.1, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(diag_decompose=True, gnn_layer_num=1, k=1)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    elif args.mode == 'decode':
        decode(args)
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
    with open(log_path, 'w', encoding='utf-8') as fout:
        fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized = True
    else:
        use_contextualized = False
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        dataset = LMGraphRelationNetDataLoader(args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=(device, device),
                                               model_name=args.encoder,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids, use_contextualized=use_contextualized,
                                               train_embs_path=args.train_embs, dev_embs_path=args.dev_embs, test_embs_path=args.test_embs,
                                               subsample=args.subsample, format=args.format)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        lstm_config = get_lstm_config_from_args(args)
        model = LMGraphRelationNet(args.encoder, k=args.k, n_type=3, n_basis=args.num_basis, n_layer=args.gnn_layer_num,
                                   diag_decompose=args.diag_decompose, n_concept=concept_num,
                                   n_relation=args.num_relation, concept_dim=args.gnn_dim,
                                   concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                                   n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                   att_dim=args.att_dim, att_layer_num=args.att_layer_num,
                                   p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                   pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                   ablation=args.ablation, init_range=args.init_range,
                                   eps=args.eps, use_contextualized=use_contextualized,
                                   do_init_rn=args.init_rn, do_init_identity=args.init_identity, encoder_config=lstm_config)
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.fix_trans:
        no_decay.append('trans_scores')
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = ConstantLRSchedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)

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

    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
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
            model.train()
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)

                    if args.loss == 'margin_rank':
                        num_choice = logits.size(1)
                        flat_logits = logits.view(-1)
                        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
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
                scheduler.step()
                optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
            print('-' * 71)
            print('| step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(global_step, dev_acc, test_acc))
            print('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                torch.save([model, args], model_path)
                print(f'model saved to {model_path}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at epoch {})'.format(best_dev_acc, best_dev_epoch))
    print('final test acc: {:.4f}'.format(final_test_acc))
    print()


def eval(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    use_contextualized = 'lm' in old_args.ent_emb
    dataset = LMGraphRelationNetDataLoader(old_args.train_statements, old_args.train_adj,
                                           old_args.dev_statements, old_args.dev_adj,
                                           old_args.test_statements, old_args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=(device, device),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=old_args.inhouse, inhouse_train_qids_path=old_args.inhouse_train_qids, use_contextualized=use_contextualized,
                                           train_embs_path=old_args.train_embs, dev_embs_path=old_args.dev_embs, test_embs_path=old_args.test_embs,
                                           subsample=old_args.subsample, format=old_args.format)

    print()
    print("***** runing evaluation *****")
    print(f'| dataset: {old_args.dataset} | num_dev: {dataset.dev_size()} | num_test: {dataset.test_size()} | save_dir: {args.save_dir} |')
    dev_acc = evaluate_accuracy(dataset.dev(), model)
    test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() else 0.0
    print("***** evaluation done *****")
    print()
    print(f'| dev_accuracy: {dev_acc} | test_acc: {test_acc} |')


def pred(args):
    raise NotImplementedError()


def decode(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    statement_dic = {}
    for statement_path in (old_args.train_statements, old_args.dev_statements, old_args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb
    dataset = LMGraphRelationNetDataLoader(old_args.train_statements, old_args.train_adj,
                                           old_args.dev_statements, old_args.dev_adj,
                                           old_args.test_statements, old_args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=(device, device),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=old_args.inhouse, inhouse_train_qids_path=old_args.inhouse_train_qids, use_contextualized=use_contextualized,
                                           train_embs_path=old_args.train_embs, dev_embs_path=old_args.dev_embs, test_embs_path=old_args.test_embs,
                                           subsample=old_args.subsample, format=old_args.format)

    with open(args.cpnet_vocab_path, 'r', encoding='utf-8') as fin:
        id2concept = [w.strip() for w in fin]

    def path_ids_to_text(path_ids):
        assert len(path_ids) % 2 == 1
        res = []
        for p in range(len(path_ids)):
            if p % 2 == 0:  # entity
                res.append(id2concept[path_ids[p].item()])
            else:  # relationi
                rid = path_ids[p].item()
                if rid < len(merged_relations):
                    res.append('<--[{}]---'.format(merged_relations[rid]))
                else:
                    res.append('---[{}]--->'.format(merged_relations[rid - len(merged_relations)]))
        return ' '.join(res)

    print()
    print("***** decoding *****")
    print(f'| dataset: {old_args.dataset} | num_dev: {dataset.dev_size()} | num_test: {dataset.test_size()} | save_dir: {args.save_dir} |')
    model.eval()
    for eval_set, filename in zip([dataset.dev(), dataset.test()], ['decode_dev.txt', 'decode_test.txt']):
        outputs = []
        with torch.no_grad():
            for qids, labels, *input_data in tqdm(eval_set):
                logits, path_ids, path_lengths = model.decode(*input_data)
                predictions = logits.argmax(1)
                for i, (qid, label, pred) in enumerate(zip(qids, labels, predictions)):
                    outputs.append('*' * 60)
                    outputs.append('id: {}'.format(qid))
                    outputs.append('question: {}'.format(statement_dic[qid]['question']))
                    outputs.append('answer: {}'.format(statement_dic[qid]['answers'][label.item()]))
                    outputs.append('prediction: {}'.format(statement_dic[qid]['answers'][pred.item()]))
                    for j, answer in enumerate(statement_dic[qid]['answers']):
                        path = path_ids[i, j, :path_lengths[i, j]]
                        outputs.append('{:25} {}'.format('[{}. {}]{}{}'.format(chr(ord('A') + j),
                                                                               answer,
                                                                               '*' if j == label else '',
                                                                               '^' if j == pred else ''),
                                                         path_ids_to_text(path)))
        output_path = os.path.join(args.save_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as fout:
            for line in outputs:
                fout.write(line + '\n')
        print(f'outputs saved to {output_path}')
    print("***** done *****")
    print()


if __name__ == '__main__':
    main()
