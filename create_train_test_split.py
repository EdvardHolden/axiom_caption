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

parser.add_argument('--fof_only', default=False, action='store_true',
                    help='Uses only FOF proofs if specified')
parser.add_argument('--random_state', default=7, type=int,
                    help='Set random set for the splitting')

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

    # Extract ids from axioms
    ids = list(axioms.keys())

    # If set only use FOF format
    if args.fof_only:
        print("Restricting problems to FOF")
        prover, competition = args.axiom_file.split('/')[-1].split('.')[0].split('_')
        # Open the meta
        with open(f'data/raw/{competition}_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        # Get problems with +1
        # TODO might change back to use problem id
        fof_problems = [prob for prob, m in meta[prover].items() if m['version'] and '+' in m['version']]
        print("Number of FOF problems found: ", len(fof_problems))

        # TODO this is a mess and will be much better when combining the data stuff!
        ids = sorted(set(ids).intersection(set(fof_problems)))
        print("Number of problems after filtering: ", len(ids))

    # Compute splits
    train_id, test_id = train_test_split(ids, shuffle=True, train_size=args.train_size, random_state=args.random_state)
    train_id, val_id = train_test_split(train_id, shuffle=True, test_size=args.val_size, random_state=args.random_state)

    # Report stats
    print(f'Resulting datasets train:{len(train_id)}, test:{len(test_id)}, val:{len(val_id)}')

    # Create directory
    # TODO this could be improved
    dir_path = 'data/' + args.axiom_file.split("/")[-1][:-4]
    if args.fof_only:
        dir_path += '_fof'
    dir_path += '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f'Saving sets to {dir_path}')

    # Save sets to path
    save_set(dir_path + 'train.txt', train_id)
    save_set(dir_path + 'test.txt', test_id)
    save_set(dir_path + 'val.txt', val_id)


if __name__ == "__main__":
    main()
