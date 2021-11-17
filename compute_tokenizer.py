import argparse
import json
from keras.preprocessing.text import Tokenizer
from dataset import load_clean_descriptions, load_ids
import config
import os

parser = argparse.ArgumentParser()
parser.add_argument('--id_file', default=config.train_id_file,
                    help="File containing the ids used to construct the tokenizer")
parser.add_argument('--proof_data', default=config.proof_data,
                    help="File containing the proof data")
parser.add_argument('--vocab_word_limit', default=None,
                    help="Number of top K words to include in the vocabulary. None for all words")


def create_tokenizer(descriptions, vocab_word_limit):
    lines = list(descriptions.values())
    tokenizer = Tokenizer(num_words=vocab_word_limit,
                          filters='',
                          split=config.TOKEN_DELIMITER,
                          oov_token=config.TOKEN_OOV)
    tokenizer.fit_on_texts(lines)
    return tokenizer


if __name__ == "__main__":

    args = parser.parse_args()

    train_descriptions = load_clean_descriptions(
        args.proof_data, load_ids(args.id_file))
    tokenizer = create_tokenizer(train_descriptions, args.vocab_word_limit)

    # Add padding token
    tokenizer.word_index[config.TOKEN_PAD] = 0
    tokenizer.index_word[0] = config.TOKEN_PAD

    print('Vocabulary Size: {0}'.format(len(tokenizer.word_index) + 1))

    # Save the tokenizer
    save_path = os.path.join(os.path.dirname(args.id_file), 'tokenizer.json')
    with open(save_path, 'w') as f:
        json.dump(tokenizer.to_json(), f)
    print("Saved tokenizer to: ", save_path)
