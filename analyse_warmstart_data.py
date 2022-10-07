import numpy as np
import itertools
from collections import Counter

from dataset import load_ids, load_warmstart_data, load_tokenizer
from enum_types import AxiomOrder

tokenizer_path = "data/deepmath/tokenizer_axioms_train_6000.json"
warmstart_data = "generated_problems/analysis/output_original_unquoted_sine_1_1"

#id_path = "data/deepmath/debug.txt"
id_path = "data/deepmath/all.txt"

def main():

    # Load ids
    ids = load_ids(id_path)

    # Load tokenizer
    caption_tokenizer, _ = load_tokenizer(tokenizer_path)

    # Load warmstart data
    data = load_warmstart_data(ids, warmstart_data, caption_tokenizer, AxiomOrder.ORIGINAL, True, None, workers=8)

    # Remove start token for easier analysis
    data = {k: v[1:] for k, v in data.items()}

    # Analysis
    print()
    print("# " * 20)
    print()
    no_tokens = [len(v) for v in data.values()]

    # Total number of problems
    print("Number of problems: ", len(data))
    print("Problems with no warmstart tokens: ", sum(1 for v in data.values() if len(v) == 0))
    print("Max number of tokens: ", max(no_tokens))
    print(f"Average: {np.mean(no_tokens):.2f}")
    print(f"Median: {np.median(no_tokens):.2f}")



    print()
    n = 10
    print(f"Top {n} most used axioms")
    values_list = list(itertools.chain.from_iterable(data.values()))
    most_common = Counter(values_list).most_common(n)
    for ax, count in most_common:
        print(f"{count:3} : {caption_tokenizer.index_word[ax]}")


if __name__ == "__main__":
    main()
