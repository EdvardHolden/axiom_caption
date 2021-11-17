import pickle
import tensorflow as tf
import os
import json
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences

import config


def get_tokenizer(tokenizer_path, verbose=0):
    # Function for loading the tokenizer from path
    with open(tokenizer_path) as f:
        tokenizer = tokenizer_from_json(json.load(f))
    vocab_size = len(tokenizer.word_index)
    if verbose > 0:
        print('Vocabulary Size: %d' % vocab_size)
    return tokenizer, vocab_size


def load_doc(filename):
    # Return the text contained in a document
    with open(filename, 'r') as file:
        text = file.read()
    return text


def load_ids(filename):
    # Load the image ids of a file
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return sorted(dataset)


def load_clean_descriptions(filename, ids, order):
    # Load the descriptions of the images in the set and append start/end tokens
    with open(filename, 'rb') as f:
        proof_data = pickle.load(f)

    descriptions = dict()
    for data in proof_data.values():
        # skip images not in the set
        if data['version'] in ids:

            # Extract axioms and manipulate the delimiter
            axioms = data['axioms']
            axioms = [ax.replace(config.TOKEN_DELIMITER, ' ') for ax in axioms]

            # Sort the list of axioms if set
            if order is not None:
                if order == 'lexicographic':
                    axioms = sorted(axioms)
                elif order == 'length':
                    axioms = sorted(axioms, key=len)
                else:
                    raise ValueError(f'Incorrect value given for order: \'{order}\'')

            # Build the caption string and save in dict
            descriptions[data['version']] = f'{config.TOKEN_START}{config.TOKEN_DELIMITER}' \
                                            + f'{config.TOKEN_DELIMITER}'.join(axioms) \
                                            + f'{config.TOKEN_DELIMITER}{config.TOKEN_END}'

    return descriptions


def load_photo_features(filename, dataset):
    # Load the img features of the photos in the set
    with open(filename, 'rb') as f:
        all_features = pickle.load(f)
    # FIXME Adding the .p extension in the name for now - should be made more uniform
    features = {k: all_features[k + '.p'] for k in dataset}
    return features


def get_dataset(image_ids, image_descriptions, image_features, tokenizer=None,
                max_cap_len=None, batch_size=config.BATCH_SIZE, order=None):

    # Load the necessary data for the id set
    ids = load_ids(image_ids)
    captions = load_clean_descriptions(image_descriptions, ids, order=order)
    img_features = load_photo_features(image_features, ids)

    # Compute the longest caption if value not provided
    if max_cap_len is None:
        max_cap_len = max(len(s.split(config.TOKEN_DELIMITER)) for s in list(captions.values()))

    # Cheat - for now we just extract and store in memory
    captions = [captions[i] for i in ids]
    img_features = [img_features[i] for i in ids]

    # Tokenize the sentences if provided
    if tokenizer is not None:
        # Tokenize captions
        captions = tokenizer.texts_to_sequences(captions)
        # Pad the tokenised captions
        captions = pad_sequences(captions, maxlen=max_cap_len, padding='post')

    # Make dataset from data lists
    dataset = tf.data.Dataset.from_tensor_slices((img_features, captions))

    # Shuffle and batch
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Return the made dataset
    return dataset, max_cap_len


def main():

    # Set order for testing purposes
    #order = None
    #order = 'lexicographic'
    order = 'length'

    # Function for testing the dataset creation
    tokenizer_path = os.path.join(os.path.dirname(config.train_id_file), 'tokenizer.json')
    tokenizer, _ = get_tokenizer(tokenizer_path)

    train_data, max_len_train = get_dataset(config.train_id_file, config.proof_data,
                                            config.problem_features, tokenizer=tokenizer,
                                            order=order)

    test_data, max_len_test = get_dataset(config.test_id_file, config.proof_data,
                                          config.problem_features, tokenizer=tokenizer,
                                          max_cap_len=max_len_train, order=order)

    print(train_data)
    print(test_data)
    print("Max length: ", max_len_train)
    assert max_len_test == max_len_train


if __name__ == "__main__":
    main()
