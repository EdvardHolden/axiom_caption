import argparse
import json
from keras.preprocessing.text import Tokenizer
from dataset import load_clean_descriptions, load_clean_conjectures, load_ids
import config
import os
from enum import Enum
import re
from pathlib import Path


def get_compute_tokenizer_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id_file",
        default=config.train_id_file,
        help="File containing the ids used to construct the tokenizer",
    )
    parser.add_argument(
        "--tokenizer_data_path",
        default=config.proof_data,
        help="File containing the proof|conjecture|text data",
    )
    parser.add_argument(
        "--vocab_word_limit",
        default=None,
        type=int,
        help="Number of top K words to include in the vocabulary. None for all words",
    )
    parser.add_argument(
        "--tokenizer_mode",
        default="axioms",
        choices=["axioms", "words", "conj_char", "conj_word"],
        help="Set preprocessing based on natural language, conjecture or axioms",
    )
    return parser


class TokenizerMode(Enum):
    """
    Helper class for setting the parameters of the tokenizer.
    """

    AXIOMS = "axioms"
    WORDS = "words"
    CONJ_CHAR = "conj_char"
    CONJ_WORD = "conj_word"

    def __str__(self):
        return self.value


def create_tokenizer(descriptions, vocab_word_limit, tokenizer_mode):
    lines = list(descriptions.values())
    if tokenizer_mode is TokenizerMode.AXIOMS:
        tokenizer = Tokenizer(
            lower=False,
            num_words=vocab_word_limit,
            filters="",
            split=config.TOKEN_DELIMITER,
            oov_token=config.TOKEN_OOV,
        )
    elif tokenizer_mode is TokenizerMode.WORDS:
        tokenizer = Tokenizer(
            lower=True,
            num_words=vocab_word_limit,
            filters='!"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~',
            split=config.TOKEN_DELIMITER,
            oov_token=config.TOKEN_OOV,
        )
    elif tokenizer_mode is TokenizerMode.CONJ_CHAR:
        tokenizer = Tokenizer(
            lower=False,
            num_words=vocab_word_limit,
            filters=" ",  # Filter whitespace
            char_level=True,
            # split=config.TOKEN_DELIMITER,
            oov_token=config.TOKEN_OOV,
        )
    elif tokenizer_mode is TokenizerMode.CONJ_WORD:
        # Same as character but all each symbol is a single token
        # Ensure space around the parenthasis for correct parsing
        lines = [re.sub("(,|\[|\]|\(|\))", r" \1 ", line) for line in lines]
        tokenizer = Tokenizer(
            filters=".",  # Remove . that might be left at the end
            num_words=vocab_word_limit,
            lower=False,
            split=" ",  # Split on all whitespace, should now all be words
            oov_token=config.TOKEN_OOV,
            char_level=False,
        )

    else:
        raise ValueError(f"Unrecognized tokenizer mode for intialising the tokenizer: {tokenizer_mode}")

    tokenizer.fit_on_texts(lines)
    return tokenizer


def get_tokenizer_save_path(dest, id_file, tokenizer_mode, vocab_word_limit) -> str:

    # Convert None to all for ebtter name representation
    if vocab_word_limit is None:
        vocab_word_limit = "all"

    # Save the tokenizer
    save_path = os.path.join(dest, f"tokenizer_{id_file}_{tokenizer_mode}_{vocab_word_limit}.json")

    return save_path


def save_tokenizer(tokenizer, save_path):

    with open(save_path, "w") as f:
        json.dump(tokenizer.to_json(), f)
    print("Saved tokenizer to: ", save_path)


def main(tokenizer_data_path, id_file, tokenizer_mode, vocab_word_limit):

    # Convert the mode to enum
    if not isinstance(tokenizer_mode, TokenizerMode):
        tokenizer_mode = TokenizerMode(tokenizer_mode)

    if tokenizer_mode is TokenizerMode.AXIOMS or tokenizer_mode is TokenizerMode.WORDS:
        # Load clean descriptions
        tokenizer_data = load_clean_descriptions(
            tokenizer_data_path, load_ids(id_file), order=None
        )  # The order is irrelevant for this purpose
    elif tokenizer_mode is TokenizerMode.CONJ_WORD or tokenizer_mode is TokenizerMode.CONJ_CHAR:
        # Load clean conejctures
        tokenizer_data = load_clean_conjectures(tokenizer_data_path, load_ids(id_file))
    else:
        raise ValueError(f"Unrecognized tokenizer mode for loading text: {tokenizer_mode}")

    # Initialise and fit the tokenizer
    tokenizer = create_tokenizer(tokenizer_data, vocab_word_limit, tokenizer_mode)

    # Add padding token
    tokenizer.word_index[config.TOKEN_PAD] = 0
    tokenizer.index_word[0] = config.TOKEN_PAD

    vocab_size = tokenizer.get_config()["num_words"]
    if vocab_size is None:
        vocab_size = len(tokenizer.word_index)
    print(f"Vocabulary Size: {vocab_size}")

    # Get dest string based on the params
    save_path = get_tokenizer_save_path(
        os.path.dirname(id_file),
        Path(id_file).stem,
        tokenizer_mode,
        vocab_word_limit,
    )

    # Save the tokenizer
    save_tokenizer(tokenizer, save_path)


def run_main():

    # Get the parser and parse the arguments
    parser = get_compute_tokenizer_parser()
    args = parser.parse_args()

    # Run main with the arguments
    main(**vars(args))


if __name__ == "__main__":
    run_main()
