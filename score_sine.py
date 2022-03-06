import argparse
import re
import os
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

import generate_problems
from model import get_model_params
from model import load_model
from generate_problems import (
    get_problems_from_path,
    load_and_process_problem,
    sine_process,
    extract_rare_axioms,
    compute_caption,
)
from evaluate import jaccard_score_np, coverage_score_np
from dataset import get_tokenizer
from dataset import load_photo_features


parser = argparse.ArgumentParser()
# parser.add_argument("--sine_sd", default=None)
# parser.add_argument("--sine_st", default=None)
parser.add_argument("--model_dir", default=None, help="Path to sequence model, used if uncluding predictions")
# experiments/model_size/attention_False_axiom_order_length_batch_norm_False_dropout_rate_0.1_embedding_size_50_learning_rate_0.001_model_type_inject_no_dense_units_512_no_rnn_units_32_normalize_True_rnn_type_lstm
parser.add_argument("--include_rare_axioms", default=False, action="store_true", help="Include rare axioms")
parser.add_argument(
    "--feature_path",
    default="data/embeddings/deepmath/graph_features_deepmath_premise.pkl",
    help="Path to the problem embeddings",
)
parser.add_argument(
    "--number_of_samples",
    type=int,
    default=None,
    help="Number of samples to use for computing the score (None for all)",
)


NO_WORKERS = 8

clause_file_re = b"file\('(\/|\w)*',(\w*)\)\)."
positive_clause_name_problem_re = b"^\+ fof\((\w+), axiom"  # only positive axioms
clause_name_problem_re = "^fof\((\w+), axiom"


# Wou ld be good with multiprocessing on the problems!
DATASET_PATH = "/home/eholden/gnn-entailment-caption/nndata/"

ST_VALUES = [1, 2, 3]
SD_VALUES = [0, 1, 2, 3, 4]


def get_sine_clause_names(prob):
    res = re.findall(clause_file_re, prob)
    res = [r[1] for r in res]
    return res


def get_original_clause_names(prob):
    res = re.findall(positive_clause_name_problem_re, prob, flags=re.MULTILINE)
    return res


def get_clause_names(prob):
    res = re.findall(clause_name_problem_re, prob, flags=re.MULTILINE)
    return res


def sine_score_problem(prob_path, sine_st, sine_sd, selected_axioms):

    prob = load_and_process_problem(prob_path)
    prob_processed = sine_process(prob, sine_st=sine_st, sine_sd=sine_sd)
    del prob

    # Extract the clause names from the sine processed problem
    sine_names = get_sine_clause_names(prob_processed)

    # Extract the clause names from the original problem
    with open(DATASET_PATH + Path(prob_path).stem, "rb") as f:  # Need to open the original dataset file
        prob_original = f.read()
    prob_names = get_original_clause_names(prob_original)
    del prob_original

    if len(prob_names) == 0:
        print(f"ERROR: No clause names for problem {prob_path}", file=sys.stderr)

    # Add selected axioms to the set of sine names
    sine_names = set(sine_names).union(selected_axioms)

    # Compute the scores - no avg as it is single entry anyways
    jaccard = jaccard_score_np([prob_names], [sine_names], avg=False)[0]
    coverage = coverage_score_np([prob_names], [sine_names], avg=False)[0]

    # Return result
    return Path(prob_path).stem, {"jaccard": jaccard, "coverage": coverage}


def sine_score_set(problem_paths, sine_st, sine_sd, selected_axioms_dict):

    # Iterate over each problem and compute their SiNe representation and their scores - include the set of selected axioms
    map_args = [
        (prob_path, sine_st, sine_sd, selected_axioms_dict.get(Path(prob_path).stem, set()))
        for prob_path in problem_paths
    ]
    pool = Pool(NO_WORKERS)
    res = pool.starmap(sine_score_problem, map_args)
    pool.close()
    pool.join()

    # Set the socre dict
    scores = {prob: s for prob, s in res}

    avg_jaccard = np.average([v["jaccard"] for v in scores.values()])
    avg_coverage = np.average([v["coverage"] for v in scores.values()])

    return avg_jaccard, avg_coverage


def print_results_table(results):

    # Function for printing table line
    def print_line():
        print("-" * (4 + (14 * len(ST_VALUES))))

    # Pretty print our results
    print("(Jaccard , Coverage)")
    print("   |" + "|".join([f"    {st:>4}     " for st in ST_VALUES]) + "|")
    print_line()
    for sd, res in zip(SD_VALUES, results):
        print(f" {sd} |" + "|".join([f" {r[0]:>5.2f} {r[1]:>5.2f} " for r in res]) + "|")
        print_line()


def get_rare_axioms(prob_path, tokenizer):
    prob = load_and_process_problem(prob_path)
    rare_axiom_clauses = extract_rare_axioms(tokenizer, prob)

    # Extract the clause names
    rare_axiom_names = get_clause_names("\n".join(rare_axiom_clauses))
    assert len(rare_axiom_names) == len(rare_axiom_clauses)

    # Convert to bytes
    rare_axiom_names = {ax.encode() for ax in rare_axiom_names}
    return Path(prob_path).stem, rare_axiom_names


def main():
    # Parse input arguments
    args = parser.parse_args()

    # Get path to all problems
    problem_paths = get_problems_from_path(DATASET_PATH, limit=args.number_of_samples)

    # We compute the sequence problems first as it is unaffected by the sine parameters.
    # Load tokenizer if needed
    # TODO maybe split this and make it into separate functions?
    selected_axioms_dict = {}
    if args.model_dir is not None or args.include_rare_axioms:
        tokenizer, _ = get_tokenizer("data/deepmath/tokenizer.json")

        # Include rare axioms if set
        rare_axioms = {}
        if args.include_rare_axioms:
            print("Extracting rare axioms")
            map_args = [(prob_path, tokenizer) for prob_path in problem_paths]
            pool = Pool(NO_WORKERS)
            res = pool.starmap(get_rare_axioms, map_args)
            pool.close()
            pool.join()
            rare_axioms = {prob: s for prob, s in res}

        # Compute axioms with model if set
        sequence_axioms = {}
        if args.model_dir is not None:
            print("Predicting axioms")
            problem_features = load_photo_features(args.feature_path, [Path(p).stem for p in problem_paths])
            model_params = get_model_params(args.model_dir)
            model_dir = os.path.join(args.model_dir, "ckpt_dir")
            model = load_model(model_dir)
            model.no_rnn_units = model_params.no_rnn_units

            for prob_path in problem_paths:
                axiom_caption = compute_caption(tokenizer, model, problem_features[Path(prob_path).stem])
                # Extract the clause names
                axiom_caption = get_clause_names("\n".join(axiom_caption))
                # assert len(rare_axiom_names) == len(rare_axiom_clauses)

                axiom_caption = [ax.encode() for ax in axiom_caption]  # Ensure axioms are bytes

                sequence_axioms[Path(prob_path).stem] = axiom_caption

        # Join the two dicts
        for prob_path in problem_paths:
            prob = Path(prob_path).stem
            selected_axioms_dict[prob] = rare_axioms.get(prob, set()).union(sequence_axioms.get(prob, set()))

    # Initialise results matrix
    results = [[-1] * len(ST_VALUES)] * len(SD_VALUES)

    # Compute results
    for nt, st in enumerate(ST_VALUES):
        for nd, sd in enumerate(SD_VALUES):
            print(f"sd:{sd} st{st}")
            res = sine_score_set(problem_paths, st, sd, selected_axioms_dict)
            results[nd][nt] = res

    print_results_table(results)


if __name__ == "__main__":
    main()
    # Extension: segment on train test?
