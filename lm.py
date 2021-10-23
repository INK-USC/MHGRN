import random

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import WarmupLinearSchedule, WarmupConstantSchedule, ConstantLRSchedule

from modeling.modeling_lm import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils.utils import *


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'extract', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/{args.dataset}.{args.encoder}.lm/', help='model output directory')
    parser.add_argument('-ckpt', '--from_checkpoint', default=None, help='load from a checkpoint')
    parser.add_argument('--subsample', default=1.0, type=float)

    # optimization
    parser.add_argument('-ebs', "--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument('-mbs', '--mini_batch_size', type=int, default=2)

    # feature extraction (used only in extract mode)
    parser.add_argument('--layer_id', default=-2, type=int, help='hidden states of layer to extract')
    args, _ = parser.parse_known_args()
    parser.add_argument('--train_output', default=f'./data/{args.dataset}/bert/{args.dataset}.train.{args.encoder}.layer{args.layer_id}.npy')
    parser.add_argument('--dev_output', default=f'./data/{args.dataset}/bert/{args.dataset}.dev.{args.encoder}.layer{args.layer_id}.npy')
    parser.add_argument('--test_output', default=f'./data/{args.dataset}/bert/{args.dataset}.test.{args.encoder}.layer{args.layer_id}.npy')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'extract':
        extract(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)


def train(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    model_path = os.path.join(args.save_dir, 'model.pt')
    check_path(model_path)

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    dataset = LMDataLoader(args.train_statements, args.dev_statements, args.test_statements,
                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                           model_name=args.encoder,
                           max_seq_length=args.max_seq_len,
                           is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids, subsample=args.subsample,
                           format=args.format)

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################

    lstm_config = get_lstm_config_from_args(args)
    model = LMForMultipleChoice(args.encoder, from_checkpoint=args.from_checkpoint, encoder_config=lstm_config)

    try:
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'lr': args.encoder_lr, 'weight_decay': 0.0}
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = ConstantLRSchedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('***** running training *****')
    print(f'| batch_size: {args.batch_size} | num_epochs: {args.n_epochs} | num_train: {dataset.train_size()} |'
          f' num_dev: {dataset.dev_size()} | num_test: {dataset.test_size()}')

    global_step = 0
    best_dev_acc = 0
    best_dev_epoch = 0
    final_test_acc = 0
    try:
        for epoch in range(int(args.n_epochs)):
            model.train()
            tqdm_bar = tqdm(dataset.train(), desc="Training")
            for qids, labels, *input_data in tqdm_bar:
                optimizer.zero_grad()
                batch_loss = 0
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    logits = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
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
                    batch_loss += loss.item()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                tqdm_bar.desc = "loss: {:.2e}  lr: {:.2e}".format(batch_loss, scheduler.get_lr()[0])
                global_step += 1

            model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() > 0 else 0.0
            if dev_acc > best_dev_acc:
                final_test_acc = test_acc
                best_dev_acc = dev_acc
                best_dev_epoch = epoch
                torch.save([model, args], model_path)
            print('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch, dev_acc, test_acc))
            if epoch - best_dev_epoch >= args.max_epochs_before_stop:
                break
    except (KeyboardInterrupt, RuntimeError) as e:
        print(e)

    print('***** training ends *****')
    print()
    print('training ends in {} steps'.format(global_step))
    print('best dev acc: {:.4f} (at epoch {})'.format(best_dev_acc, best_dev_epoch))
    print('final test acc: {:.4f}'.format(final_test_acc))
    print()


def extract(args):  # Note: extract mode ALWAYS use the official split
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    for split in ('train', 'dev') if old_args.test_statements is None else ('train', 'dev', 'test'):
        setattr(args, f'{split}_output', getattr(args, f'{split}_output').format(dataset=old_args.dataset,
                                                                                 setting=('inhouse' if old_args.inhouse else 'official'),
                                                                                 encoder_name=old_args.encoder_name,
                                                                                 layer_id=args.layer_id))

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()
    encoder = model.encoder

    dataset = LMDataLoader(old_args.train_statements, old_args.dev_statements, old_args.test_statements,
                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                           model_name=old_args.encoder, max_seq_length=old_args.max_seq_len,
                           is_inhouse=False, inhouse_train_qids_path=old_args.inhouse_train_qids)

    print()
    print("***** extracting sentence vectors *****")
    print(f'| dataset: {old_args.dataset} | layer_id: {args.layer_id} | eval_batch_size: {args.eval_batch_size} | train_output: {args.train_output} |')
    with torch.no_grad():
        for output_path, data_loader in [(args.train_output, dataset.train_eval()),
                                         (args.dev_output, dataset.dev())] + ([args.test_output, dataset.test()] if dataset.test_size() > 0 else []):
            sent_vecs = []
            for qids, labels, *input_data in tqdm(data_loader, desc='Extracting'):
                input_data = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in input_data]
                batch_sent_vecs, _ = encoder(*input_data, layer_id=args.layer_id)
                sent_vecs.append(batch_sent_vecs.cpu())
            sent_vecs = torch.cat(sent_vecs, 0).numpy()
            np.save(output_path, sent_vecs)
    print('***** extraction done *****')


def eval(args):
    model_path = os.path.join(args.save_dir, 'model.pt')
    model, old_args = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    dataset = LMDataLoader(old_args.train_statements, old_args.dev_statements, old_args.test_statements,
                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                           model_name=old_args.encoder, max_seq_length=old_args.max_seq_len,
                           is_inhouse=old_args.inhouse, inhouse_train_qids_path=old_args.inhouse_train_qids)

    print()
    print("***** runing evaluation *****")
    print(f'| dataset: {old_args.dataset} | num_dev: {dataset.dev_size()} | num_test: {dataset.test_size()} | save_dir: {args.save_dir} |')
    dev_acc = evaluate_accuracy(dataset.dev(), model)
    test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() else 0.0
    print("***** evaluation done *****")
    print()
    print(f'| dev_accuracy: {dev_acc} | test_acc: {test_acc} |')


def pred(args):  # Note: pred mode ALWAYS uses the official split
    dev_pred_path = os.path.join(args.save_dir, 'predictions_dev.csv')
    test_pred_path = os.path.join(args.save_dir, 'predictions_test.csv')
    model_path = os.path.join(args.save_dir, 'model.pt')
    old_args, model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)
    model.eval()

    dataset = LMDataLoader(old_args.train_statements, old_args.dev_statements, old_args.test_statements,
                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                           model_name=old_args.encoder, max_seq_length=old_args.max_seq_len,
                           is_inhouse=False, inhouse_train_qids_path=old_args.inhouse_train_qids)

    print("***** generating model predictions *****")
    print(f'| dataset: {old_args.dataset} | save_dir: {args.save_dir} |')

    for output_path, data_loader in [(dev_pred_path, dataset.dev())] + ([(test_pred_path, dataset.test())] if dataset.test_size() > 0 else []):
        with torch.no_grad(), open(output_path, 'w', encoding='utf-8') as fout:
            for qids, labels, *input_data in tqdm(data_loader):
                logits = model(*input_data)
                for qid, pred_label in zip(qids, logits.argmax(1)):
                    fout.write('{},{}\n'.format(qid, chr(ord('A') + pred_label.item())))
        print(f'predictions saved to {output_path}')
    print('***** prediction done *****')


if __name__ == '__main__':
    main()
