import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from keras.preprocessing.text import text_to_word_sequence

from process_problem import load_and_process_problem, get_problems_from_path
from evaluate import jaccard_score_np, coverage_score_np
from dataset import load_ids, load_tokenizer

import sys
sys.path.insert(0,'online')
from online.get_scores import get_solved_problem_name_time


# The direcotry to the proof axioms
PROOF_AXIOMS = 'generated_problems/analysis/output_original_positive_axioms/'
PROOF_AXIOMS_UNQUOTED = 'generated_problems/analysis/output_original_unquoted_positive_axioms/'

CONFIGS = {115498: 'generated_problems/analysis/output_original_ideal', # Upper bound
           115507: 'generated_problems/analysis/output_original_clean/', # Raw merged problem
           115555: 'generated_problems/analysis/output_original_sine_1_1/',
           115556: 'generated_problems/analysis/output_original_sine_3_0/',
           115591: 'generated_problems/analysis/output_original_caption/',
           #115633: 'generated_problems/analysis/output_original_caption_sine_1_1/', TODO need to recompute these problems
           115634: 'generated_problems/analysis/output_original_caption_sine_3_0/'}

# Base config used to compute the ratio of the problem selected (output_original_clean -> clean)
BASE_CONFIG = 'clean'

# Path to the problems in the training set
TRAINING_SET_PATH = 'data/deepmath/train.txt'

# Path to tokenizer used in the training context
TOKENIZER_PATH = 'data/deepmath/tokenizer_axioms_train_6000.json'


# Make sure the result are not lost due to being truncated
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)


parser = argparse.ArgumentParser()

parser.add_argument("--print_df", default=False, action="store_true",
                    help="Print detailed stat information for each problem in each analysis case")
parser.add_argument("--debug", "-d", default=False, action="store_true",
                    help="Limit number of problems for efficient debugging")
parser.add_argument("--remove_unsolved", default=False, action="store_true",
                    help="Only looks at the solved problems in the general analysis")
args = parser.parse_args()

# Set debugging mode
if args.debug:
    LIMIT = 100
else:
    LIMIT = None


def load_generated_problem(problem_path):

    # Load generated problem
    prob = load_and_process_problem(problem_path, deepmath=False)
    if len(prob) < 1:
        # Report and skip empty problems
        print(f'Warning: empty problem {problem_path}')
        return []
    # Skip the conjecture to make a jaccard score of 1 possible
    if 'conjecture' in prob[0]:
        prob = prob[1:]

    return prob


def load_proof_axioms(problem_name, unquoted=False):

    # Load proof axioms
    if unquoted:
        proof = load_and_process_problem(os.path.join(PROOF_AXIOMS_UNQUOTED, problem_name), deepmath=False)
    else:
        proof = load_and_process_problem(os.path.join(PROOF_AXIOMS, problem_name), deepmath=False)
    # Hack: check if quoting / $distinct number axioms is added. if so, remove it
    if "a1, axiom, $distinct" in proof[-1]:
        proof = proof[:-1]

    return proof


def get_metrics(problem_paths):

    results = {}

    for problem_path in problem_paths:
        # Get the problem name
        name = Path(problem_path).stem

        proof = load_proof_axioms(name)

        prob = load_generated_problem(problem_path)

        results[name] = {"jaccard": jaccard_score_np([proof], [prob]),
                         "coverage": coverage_score_np([proof], [prob]),
                         "length": len(prob)}

    return results


def analyse_partition_metrics(config, df):

    query_columns = [f"{config}_jaccard", f"{config}_coverage", f"{config}_length", f"{config}_ratio"]

    if 'caption' in config:
        query_columns += ["tokenizer_rare", "tokenizer_predictable"]

    query_columns += [f"{config}_time"]

    # Report the statistics
    solved = df.loc[df[f"{config}_solved"]].index
    print('## Solved partition')
    print(df[query_columns].loc[solved].describe())
    print()
    print('## Unsolved partition')
    print(df[query_columns[:-1]].loc[set(df.index) - set(solved)].describe())


def get_performance_stats(exp_id):

    # Get problem ID and runtime of the solved problems in the experiment(LTB)
    res = get_solved_problem_name_time(exp_id)

    # Convert to dict
    res = {name: {"solved_time": time} for name, time in res}

    return res


def run_partition_analysis(configs, df):
    for conf in configs:
        print(f"## ## {conf}")
        analyse_partition_metrics(conf, df)
        print()
        print()


def get_in_training_set_metric(result):

    train_ids = load_ids(TRAINING_SET_PATH)
    for prob in result:
        if prob in train_ids:
            result[prob].update({"in_training": True})
        else:
            result[prob].update({"in_training": False})

    return result


def compute_rare_proportion(tokenizer, proof):

    no_rare = 0
    # If the clause is known to use, but not in the top words, it is positively rare
    for formula in proof:
        i = tokenizer.word_index.get(formula)
        if i is not None and i >= tokenizer.num_words:
            no_rare += 1

    return no_rare / len(proof)


def compute_predictable_proportion(tokenizer, proof):

    no_predictable = 0
    # If the clause is known to use, but not in the top words, it is predictable
    for formula in proof:
        i = tokenizer.word_index.get(formula)
        if i is not None and i < tokenizer.num_words:
            no_predictable += 1

    return no_predictable / len(proof)


def get_tokenizer_metrics(result):
    '''
    Compute what proportion of the proofs that are covered by the set tokenizer
    '''

    # Load tokenizer from path
    tokenizer, _ = load_tokenizer(TOKENIZER_PATH)

    for prob in result:

        # Load proof - need to make sure that these are unquoted like in the tokenizer
        proof = load_proof_axioms(prob, unquoted=True)

        # Process the formulae
        proof = [text_to_word_sequence(formula, tokenizer.filters, tokenizer.lower, tokenizer.split)[0] for formula in proof]

        # Compute rare proportion
        rare = compute_rare_proportion(tokenizer, proof)

        # Compute predictable proprtion
        predictable = compute_predictable_proportion(tokenizer, proof)

        # Update stats
        result[prob].update({"tokenizer_rare": rare,
                             "tokenizer_predictable": predictable})

    return result



def get_performance_coverage_data(CONFIGS, common_substring=''):

    result = {}
    configs = []

    # Want config_coverage, config_solved
    for exp_id, problem_dir in CONFIGS.items():

        conf = Path(problem_dir).stem.replace(common_substring, '')
        configs += [conf]

        # Load the problems from path
        problems = get_problems_from_path(problem_dir, limit=LIMIT, verbose=0)
        print('# Number of problems found:', conf, len(problems))

        # Initialise - if needed
        if len(result) == 0:
            result = {Path(p).stem: {} for p in problems}

        # Get solved status and time
        performance = get_performance_stats(exp_id)

        # Get coverage scores and add performance
        for problem_path in problems:
            name = Path(problem_path).stem
            prob = load_generated_problem(problem_path)
            proof = load_proof_axioms(name)
            result[name].update({f"{conf}_coverage": coverage_score_np([proof], [prob])})
            result[name].update({f"{conf}_jaccard": jaccard_score_np([proof], [prob])})
            result[name].update({f"{conf}_length": len(prob)})

            if name in performance:
                result[name].update({f"{conf}_solved": True})
                result[name].update({f"{conf}_time": float(performance[name]["solved_time"])})
            else:
                result[name].update({f"{conf}_solved": False})

    # For each problem, load the proof and compute the tokenizer related metrics
    result = get_tokenizer_metrics(result)

    # For each problem, check whether it was in the training set or not
    result = get_in_training_set_metric(result)

    # Create the dataframe from the dictionaries
    df = pd.DataFrame.from_dict(result).T

    # Compute the ratio of selected formulae vs original formulae (slightly skewed for caption)
    for conf in configs:
        df[f'{conf}_ratio'] = df[f'{conf}_length'].values / df[f'{BASE_CONFIG}_length'].values

    # Convert to "best" possible types - avoids representing everything as objects, which messes up .describe()
    df = df.convert_dtypes()

    return configs, df


def print_avg_stats(df, base, other):

    print(f"Avg coverage base : {np.average(df[f'{base}_coverage']):.2f}")
    print(f"Avg coverage base : {np.average(df[f'{base}_coverage']):.2f}")
    print(f"Avg jaccard  base : {np.average(df[f'{base}_jaccard']):.2f}")
    print(f"Avg length   base : {np.average(df[f'{base}_length']):.2f}")
    print(f"Avg ratio    base : {np.average(df[f'{base}_ratio']):.2f}")
    print()
    print(f"Avg coverage other : {np.average(df[f'{other}_coverage']):.2f}")
    print(f"Avg jaccard  other : {np.average(df[f'{other}_jaccard']):.2f}")
    print(f"Avg length   other : {np.average(df[f'{other}_length']):.2f}")
    print(f"Avg ratio    other : {np.average(df[f'{other}_ratio']):.2f}")

    if 'caption' in base or 'caption' in other:
        print()
        print(f"Avg tokenizer rare : {np.average(df['tokenizer_rare']):.2f}")
        print(f"Avg predictable    : {np.average(df['tokenizer_predictable']):.2f}")
        print(f"Ratio of training  : {sum(df['in_training']) / len(df):.2f}")



def print_detailed_overview(df, base, other):

    query_columns = [f"{base}_coverage", f"{base}_jaccard", f"{base}_length", f"{other}_coverage", f"{other}_jaccard", f"{other}_length"]
    if "caption" in base or "caption" in other:
        query_columns += ['tokenizer_rare', 'tokenizer_predictable', 'in_training']

    print(df[query_columns])


def print_similar_problems_overview(df, base, other):

    query_columns = [f"{base}_coverage", f"{base}_jaccard", f"{base}_length", f"{base}_time"]
    if "caption" in base or "caption" in other:
        query_columns += ['tokenizer_rare', 'tokenizer_predictable', 'in_training']

    print(df[query_columns])


def compare_versions(df, base, other):

    print(f"### Comparing {base} to {other}")
    print(f"{base} solved: {len(df.loc[df[base+'_solved']])}")
    print(f"{other} solved: {len(df.loc[df[other+'_solved']])}")

    # Compute solved by base but not by other
    diff = sorted(set(df.loc[df[f'{base}_solved']].index).difference(set(df.loc[df[f'{other}_solved']].index)))
    df_comp = df.loc[diff]

    print(f"Solved by {base} and not other: ", len(diff))
    # Need to have at least one problem to compute a meaningful average
    if len(diff) > 0:
        # Print average of main stats between the different approaches
        print_avg_stats(df_comp, base, other)

        if args.print_df:
            # Print more detailed overview and separate of dissimilar stats
            # Check whether there are any problems with similar stats
            similar = df.loc[np.isclose(list(df[f"{base}_length"]), list(df[f"{other}_length"])) & np.isclose(list(df[f"{base}_coverage"]), list(df[f"{other}_coverage"])) & np.isclose(list(df[f"{base}_jaccard"]), list(df[f"{other}_jaccard"]))].index

            # Split based on whether the values are the same
            print("## Detailed Overview:")
            if len(similar) == 0:
                print_detailed_overview(df_comp, base, other)
            else:
                print("# Similar Problems: ")
                #print_detailed_overview(df_comp.loc[sorted(set(diff).intersection(set(similar)))], base, other) FIXME
                print_similar_problems_overview(df_comp.loc[sorted(set(diff).intersection(set(similar)))], base, other)
                print()
                print("# Other Problems: ")
                print_detailed_overview(df_comp.loc[sorted(set(diff).difference(set(similar)))], base, other)
    print()


def compute_solved_set(configs, df):

    solved_all = set()
    solved_method = set()

    for conf in configs:
        print(f"# Problems solved by {conf}: {sum(df[conf + '_solved'])}")

        # Add solved problems to set
        solved_all.update(df.loc[df[conf + "_solved"]].index)

        if "ideal" not in conf:
            solved_method.update(df.loc[df[conf + "_solved"]].index)

    print()
    print("# Number of problems solved in union:", len(solved_all))
    print("# Number of problems solved in union (excluding ideal):", len(solved_method))
    print()

    # We return the set of problems excluding ideal
    return solved_method

def compute_problems_lost_by_combination(df, print_df=False):

    # Compute the set difference
    diff_problems = (set(df.loc[df['sine_3_0_solved']].index).union(set(df.loc[df['caption_solved']].index))).difference(set(df.loc[df['caption_sine_3_0_solved']].index))

    print("Number of problems lost by combining sine and captioning:", len(diff_problems))
    query_columns = ['caption_sine_3_0_coverage', 'caption_sine_3_0_jaccard', 'sine_3_0_solved', 'sine_3_0_coverage', 'sine_3_0_jaccard', 'caption_solved', 'caption_coverage', 'caption_jaccard']

    # Report general stats of the lost problems (not solved/unsolved?)
    print(df[[qc for qc in query_columns if '_solved' not in qc]].loc[diff_problems].describe())

    if len(diff_problems) > 0 and print_df:
        print(df[query_columns].loc[diff_problems])


def _get_solved_set(df, configs):

    # Compute set of solved problems
    solved = set()
    for conf in configs:
        solved.update(df.loc[df[f"{conf}_solved"]].index)
    return solved


def compute_uniquely_solved(df, configs):

    for conf in configs:
        solved_others = _get_solved_set(df, set(configs).difference(set([conf])))
        unique = set(df.loc[df[f"{conf}_solved"]].index).difference(solved_others)
        print(f"{conf:16} : {len(unique)}")


def check_combined_solved(df):

    # Get problem solved by neither caption or sine, but by the combination
    # This can be important due to the diversity of the problems solved
    solved_combination = set(df.loc[df['caption_sine_3_0_solved']].index)
    sine_not_solved = set(df.index).difference(set(df.loc[df['sine_3_0_solved']].index))
    caption_not_solved = set(df.index).difference(set(df.loc[df['caption_solved']].index))

    index = solved_combination.intersection(sine_not_solved.intersection(caption_not_solved))
    # Reduce df
    df = df.loc[sorted(index)]



    print()
    print("# Problems solved by caption_sine_3_0 but neither by caption nor sine_3_0")
    print("# Number of problems ", len(index))

    # What do I want to know??
    query_columns = ["caption_sine_3_0_coverage", "caption_sine_3_0_jaccard",
                     "sine_3_0_coverage", "sine_3_0_jaccard",
                     "caption_coverage", "caption_jaccard"]
    if len(df) > 0:
        print(df[query_columns].describe())

        if args.print_df:
            print()
            print(df[query_columns])




def main():

    print('#', args)
    print()

    # Get the data needed
    configs, df = get_performance_coverage_data(CONFIGS, common_substring='output_original_')

    # Run some basic analysis on solved/unsolved for metrics of each configuration
    print()
    print("# # # # # # # # # # # # #")
    print("# # General Analysis  # #")
    print()
    # Compute the solved set
    solved_set = compute_solved_set(configs, df)

    # Compute problems that are inquely solved by a config
    print("# Uniquely solved overall")
    compute_uniquely_solved(df, configs)
    print()
    print("# Uniquely solved excluding ideal")
    compute_uniquely_solved(df, [conf for conf in configs if 'ideal' not in conf])
    print()

    # Print stats of the solved/unsolved partition of each config
    print()
    print("# # # # # # # # # # # # #")
    print("# # Partition Analysis  # #")
    print()
    # Check if we should limit the analysis to the set f solved problems (excluding ideal)
    if args.remove_unsolved:
        print("!! Limiting analysis to problems solved by at least one config other than \"ideal\"")
        df = df.loc[solved_set]

    run_partition_analysis(configs, df)

    # Specific analysis
    print('\n\n')
    print("# # # # # # # # # # # # #")
    print("# # Specific Analysis # #")


    # Compute problem loss of combining caption and sine_3_0
    compute_problems_lost_by_combination(df, print_df=args.print_df)

    # Compare the experiments
    print()
    compare_versions(df, 'sine_1_1', 'sine_3_0')
    print()
    compare_versions(df, 'sine_3_0', 'sine_1_1')
    print()
    compare_versions(df, 'caption', 'sine_3_0')
    print()
    compare_versions(df, 'sine_3_0', 'caption')
    print()
    compare_versions(df, 'caption_sine_3_0', 'sine_3_0')
    print()
    compare_versions(df, 'caption_sine_3_0', 'caption')


    print()
    check_combined_solved(df) # FIXME poor naming


if __name__ == "__main__":
    main()
