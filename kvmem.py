import random

from tqdm import tqdm
from transformers import (ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule)

from modeling.modeling_kvmem import *
from utils.datasets import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *

DECODER_DEFAULT_LR = {'csqa': 1e-3, 'obqa': 1e-3}

def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/{args.dataset}.{args.encoder}.kvmem/', help='model output directory')

    # data
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--train_triples', default=f'./data/{args.dataset}/triples/train.triples.pk')
    parser.add_argument('--dev_triples', default=f'./data/{args.dataset}/triples/dev.triples.pk')
    parser.add_argument('--test_triples', default=f'./data/{args.dataset}/triples/test.triples.pk')

    # model architecture
    parser.add_argument('--ablation', default=None, choices=['None', 'no_kg', 'no_2hop', 'no_1hop', 'no_qa', 'no_rel', 'singlehead',
                                                             'mrloss', 'fixrel', 'fakerel', 'factor_add', 'factor_mul', 'no_2hop_qa',
                                                             'mean_pool', 'randomrel', 'encode_qas'], help='run ablation test')
    parser.add_argument('--max_triple_num', type=int, default=300, help='max num of triples')
    parser.add_argument('--gamma', type=float, default=0.5, help='ratio')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--decoder_num_layers', default=2, type=int)
    parser.add_argument('--decoder_bidirectional', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--cpt_out_dim', type=int, default=300, help='num of dimension for concepts in processing')
    parser.add_argument('--subsample', default=1.0, type=float)

    # regularization
    parser.add_argument('--d_dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--d_dropouti', type=float, default=0.1, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--d_dropoutr', type=float, default=0.1, help='dropout for rnn hidden units (0 = no dropout)')
    parser.add_argument('--d_dropouto', type=float, default=0.1, help='dropout for rnn outputs (0 = no dropout')
    parser.add_argument('--d_dropoutm', type=float, default=0.1, help='dropout for mlp hidden units (0 = no dropout')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument('--save', type=bool_flag, default=False, help='whether to save logs and models')
    parser.add_argument('--use_single_gpu', type=bool_flag, default=True)
    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)

    if args.encoder in ('lstm',):
        parser.set_defaults(unfreeze_epoch=0)

    # set dataset defaults
    parser.set_defaults(num_choice=dataset_num_choice[args.dataset],
                        inhouse=(dataset_setting[args.dataset] == 'inhouse'),
                        inhouse_train_qids=args.inhouse_train_qids.format(dataset=args.dataset),
                        save_dir=args.save_dir.format(dataset=args.dataset))
    parser.set_defaults(test_statements=None, test_triples=None)
    for split in ('train', 'dev') if args.dataset in dataset_no_test else ('train', 'dev', 'test'):
        for attribute in ('statements', 'triples'):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})

    # set ablation defaults
    if args.ablation == 'singlehead':
        parser.set_defaults(att_head_num=1)
    elif args.ablation == 'mrloss':
        parser.set_defaults(loss='margin_rank')
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
    if args.save:
        export_config(args, config_path)
        check_path(model_path)
        with open(log_path, 'w', encoding='utf-8') as fout:
            fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    cp_emb, rel_emb = [np.load(path) for path in args.ent_emb_paths], np.load(args.rel_emb_path)
    cp_emb = np.concatenate(cp_emb, axis=1)
    cp_emb = torch.tensor(cp_emb)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('num_concepts: {}, concept_dim: {}'.format(concept_num, concept_dim))
    relation_num = len(rel_emb) * 2
    print('num_relations: {}'.format(relation_num))

    try:
        device0 = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
        if args.use_single_gpu:
            device1 = device0
        else:
            device1 = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
        dataset = KVMDataLoader(train_statement_path=args.train_statements, train_triple_pk=args.train_triples,
                                dev_statement_path=args.dev_statements, dev_triple_pk=args.dev_triples,
                                test_statement_path=args.test_statements, test_triple_pk=args.test_triples,
                                concept2id_path=args.cpnet_vocab_path, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                device=(device0, device1), model_name=args.encoder, max_triple_num=args.max_triple_num,
                                max_seq_length=args.max_seq_len, is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                subsample=args.subsample, format=args.format)

        print('len(train_set): {}   len(dev_set): {}   len(test_set): {}'.format(dataset.train_size(), dataset.dev_size(), dataset.test_size()))
        print()

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################

        lstm_config = get_lstm_config_from_args(args)
        model = LMKVM(model_name=args.encoder, gamma=args.gamma,
                      concept_num=concept_num, concept_dim=args.cpt_out_dim, concept_in_dim=concept_dim, freeze_ent_emb=args.freeze_ent_emb, concept_emb=cp_emb,
                      relation_num=relation_num,
                      decoder_num_layers=args.decoder_num_layers, decoder_bidirectional=args.decoder_bidirectional,
                      decoder_input_p=args.d_dropouti, decoder_output_p=args.d_dropouto,
                      decoder_emb_p=args.d_dropoute, decoder_hidden_p=args.d_dropoutr, decoder_mlp_p=args.d_dropoutm,
                      encoder_config=lstm_config)

        if args.freeze_ent_emb:
            freeze_net(model.decoder.concept_emb)

        model.encoder.to(device0)
        model.decoder.to(device1)
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

    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    try:
        model.train()
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
            print('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, dev_acc, test_acc))
            print('-' * 71)
            if args.save:
                with open(log_path, 'a') as fout:
                    fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save:
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
    raise NotImplementedError()


def pred(args):
    raise NotImplementedError()


if __name__ == '__main__':
    main()
