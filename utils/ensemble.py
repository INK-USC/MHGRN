import argparse


def ensemble(input_files, output):
    in_files = [open(file, 'r') for file in input_files]  # list of fin handlers
    with open(output, 'w', encoding='utf-8') as fout:
        for *lines, in zip(*in_files):
            qids = [line.strip().split(',')[0] for line in lines]
            answers = [line.strip().split(',')[1] for line in lines]
            assert len(set(qids)) == 1  # assert that all answers correspond to the same example
            majority = max(sorted(list(set(answers))), key=answers.count)
            fout.write('{},{}\n'.format(qids[0], majority))
    [fin.close() for fin in in_files]
    print(f'ensembled results saved to {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ensemble', choices=['ensemble'])
    parser.add_argument('--input_files', default=['./saved_models/relpath/test_pred_res.csv',
                                                  './saved_models/relpath/test_pred_res.csv'], nargs='+')
    parser.add_argument('-o', '--output', default='./exp/ensemble_pred_test.csv')
    args = parser.parse_args()

    if args.mode == 'ensemble':
        ensemble(args.input_files, args.output)


if __name__ == '__main__':
    main()
