"""
This program computes train-test-val sets for a given axiom dictionary downloaded
from the CASC results. The IDs of each set are saved in a text file in an appropriate
directory in 'data/'.
"""
import argparse
import pickle
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--axiom_file', default='data/raw/vampire_jjt.pkl',  # from config TODO
                    help='Axiom file used to compute splits')

parser.add_argument('--min_axioms', default=1,
                    help='Min axiom count for being included in the split')
parser.add_argument('--max_axioms', default=100,
                    help='Max axiom count for being included in the split')

parser.add_argument('--train_size', default=0.8)
parser.add_argument('--val_size', default=0.1)


def save_set(path, ids):
    with open(path, 'w') as f:
        f.write('\n'.join(ids))


def main():

    args = parser.parse_args()

    # Load pickle
    with open(args.axiom_file, 'rb') as f:
        axioms = pickle.load(f)
    print(f'Loaded {len(axioms)} proofs from {args.axiom_file}')

    # Remove min proofs
    for prob in list(axioms.keys()):
        if len(axioms[prob]) < args.min_axioms or len(axioms[prob]) > args.max_axioms:
            del axioms[prob]
    print(f'{len(axioms)} axioms remaining after filtering on lengths min:{args.min_axioms} max:{args.max_axioms}')

    # Compute splits
    ids = list(axioms.keys())
    train_id, test_id = train_test_split(ids, shuffle=True, train_size=args.train_size)
    train_id, val_id = train_test_split(train_id, shuffle=True, test_size=args.val_size)

    # Report stats
    print(f'Resulting datasets train:{len(train_id)}, test:{len(test_id)}, val:{len(val_id)}')

    # Create directory
    dir_path = 'data/' + args.axiom_file.split("/")[-1][:-4] + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f'Saving sets to {dir_path}')

    # Save sets to path
    save_set(dir_path + 'train.txt', train_id)
    save_set(dir_path + 'test.txt', test_id)
    save_set(dir_path + 'val.txt', val_id)


if __name__ == "__main__":
    main()
