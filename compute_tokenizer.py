import argparse
import json
from keras.preprocessing.text import Tokenizer
from dataset import load_clean_descriptions, load_ids
import config
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--id_file", default=config.train_id_file, help="File containing the ids used to construct the tokenizer"
)
parser.add_argument("--proof_data", default=config.proof_data, help="File containing the proof data")
parser.add_argument(
    "--vocab_word_limit",
    default=None,
    type=int,
    help="Number of top K words to include in the vocabulary. None for all words",
)
parser.add_argument(
    "--tokenizer_type",
    choices=["axioms", "words"],
    help="Set preprocessing ased on natural language or axioms",
)


def create_tokenizer(descriptions, vocab_word_limit, axiom_words=True):
    lines = list(descriptions.values())
    if axiom_words:
        tokenizer = Tokenizer(
            lower=False,
            num_words=vocab_word_limit,
            filters="",
            split=config.TOKEN_DELIMITER,
            oov_token=config.TOKEN_OOV,
        )
    else:
        tokenizer = Tokenizer(
            lower=True,
            num_words=vocab_word_limit,
            filters='!"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~',
            split=config.TOKEN_DELIMITER,
            oov_token=config.TOKEN_OOV,
        )

    tokenizer.fit_on_texts(lines)
    return tokenizer


def main():
    args = parser.parse_args()

    train_descriptions = load_clean_descriptions(
        args.proof_data, load_ids(args.id_file), order=None
    )  # The order is irrelevant for this purpose

    # Check whether we are using axioms or natural language, no filtering for axioms
    axiom_words = args.tokenizer_type == "axioms"

    tokenizer = create_tokenizer(train_descriptions, args.vocab_word_limit, axiom_words=axiom_words)

    # Add padding token
    tokenizer.word_index[config.TOKEN_PAD] = 0
    tokenizer.index_word[0] = config.TOKEN_PAD

    vocab_size = tokenizer.get_config()["num_words"]
    if vocab_size is None:
        vocab_size = len(tokenizer.word_index)
    print(f"Vocabulary Size: {vocab_size}")

    # Save the tokenizer
    save_path = os.path.join(os.path.dirname(args.id_file), f"tokenizer_{args.vocab_word_limit}.json")
    with open(save_path, "w") as f:
        json.dump(tokenizer.to_json(), f)
    print("Saved tokenizer to: ", save_path)


if __name__ == "__main__":
    main()
