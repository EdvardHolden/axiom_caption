import pickle
from collections import Counter
import tensorflow as tf
import random
import os
import json
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from enum import Enum, auto

import config


# Set axiom order type
class AxiomOrder(Enum):
    ORIGINAL = auto()
    LEXICOGRAPHIC = auto()
    LENGTH = auto()
    FREQUENCY = auto()
    RANDOM = auto()


def get_tokenizer(tokenizer_path, verbose=0):
    # Function for loading the tokenizer from path
    with open(tokenizer_path) as f:
        tokenizer = tokenizer_from_json(json.load(f))
    vocab_size = tokenizer.num_words
    if verbose > 0:
        print("Vocabulary Size: %d" % vocab_size)
    return tokenizer, vocab_size


def load_doc(filename):
    # Return the text contained in a document
    with open(filename, "r") as file:
        text = file.read()
    return text


def max_caption_length(captions):
    return max(len(s.split(config.TOKEN_DELIMITER)) for s in list(captions.values()))


def compute_max_length(image_ids, image_descriptions):
    ids = load_ids(image_ids)
    captions = load_clean_descriptions(
        image_descriptions, ids, order=None
    )  # Order does not matter for this purpose
    return max_caption_length(captions)


def load_ids(filename):
    # Load the image ids of a file
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split("\n"):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split(".")[0]
        dataset.append(identifier)
    return sorted(dataset)


def load_clean_descriptions(filename, ids, order, axiom_frequency=None):
    # Load the descriptions of the images in the set and append start/end tokens
    with open(filename, "rb") as f:
        proof_data = pickle.load(f)

    descriptions = dict()
    for prob_id, data in proof_data.items():
        # If we have versions, the id of the problem is the version name
        if "version" in data:
            prob_id = data["version"]

        # Skip problems not in the ID set
        if prob_id in ids:

            # Extract axioms and manipulate the delimiter
            axioms = data["axioms"]
            axioms = [ax.replace(config.TOKEN_DELIMITER, " ") for ax in axioms]

            # TODO this should be a separate function
            # Sort the list of axioms if set
            if order is not None:
                # We are expecting and order but want to chedck that the type is correct
                # and order != AxiomOrder.original:
                if not isinstance(order, AxiomOrder):
                    raise ValueError(
                        f"Incorrect value given for order: '{order}' needs to be of type AxiomOrder"
                    )
                if order is AxiomOrder.ORIGINAL:
                    pass  # Do not change the order
                elif order is AxiomOrder.LEXICOGRAPHIC:
                    axioms = sorted(axioms)
                elif order is AxiomOrder.LENGTH:
                    axioms = sorted(axioms, key=len)
                elif order is AxiomOrder.RANDOM:
                    # Randomly mutates the list
                    random.shuffle(axioms)
                elif order is AxiomOrder.FREQUENCY:
                    if axiom_frequency is None:
                        raise ValueError("Order is set to frequency, but no frequency list is supplied")
                    axioms = sorted(axioms, key=lambda x: -axiom_frequency[x])
                else:
                    raise ValueError(f"No ordering function implemented for order: '{order}'")

            # Build the caption string and save in dict
            descriptions[prob_id] = (
                f"{config.TOKEN_START}{config.TOKEN_DELIMITER}"
                + f"{config.TOKEN_DELIMITER}".join(axioms)
                + f"{config.TOKEN_DELIMITER}{config.TOKEN_END}"
            )

    return descriptions


def load_photo_features(filename, dataset):
    # Load the img features of the photos in the set
    with open(filename, "rb") as f:
        all_features = pickle.load(f)
    # FIXME Adding the .p extension in the name for now - should be made more uniform
    if ".p" == str(all_features[list(all_features.keys())[0]][-2:]):
        features = {k: all_features[k + ".p"] for k in dataset}
    else:
        features = {k: all_features[k] for k in dataset}

    return features


def get_dataset(
    image_ids,
    image_descriptions,
    image_features,
    tokenizer=None,
    max_cap_len=None,
    batch_size=config.BATCH_SIZE,
    order=None,
    axiom_frequency=None,
    remove_unknown=False,
):

    # Load the necessary data for the id set
    ids = load_ids(image_ids)
    captions = load_clean_descriptions(image_descriptions, ids, order=order, axiom_frequency=axiom_frequency)
    img_features = load_photo_features(image_features, ids)

    # Compute the longest caption if value not provided
    if max_cap_len is None:
        max_cap_len = max_caption_length(captions)

    # Tokenize the sentences if provided
    if tokenizer is not None:
        # Tokenize each caption and store it back in the dictionary
        if remove_unknown:
            tokenizer.oov_token = None  # This skips entries that maps to oov
        for i in captions:
            captions[i] = tokenizer.texts_to_sequences([captions[i]])[0]

    # Build data lists
    caption_data = []
    feature_data = []
    for i in ids:
        # Remove captions that have reduced to start+end tokens due to unknown removal
        if not remove_unknown or (remove_unknown and len(captions[i]) > 2):
            caption_data.append(captions[i])
            feature_data.append(img_features[i])

    # Delete dict variables to save memory
    del captions
    del img_features

    # Pad the tokenised captions
    if tokenizer:
        caption_data = pad_sequences(caption_data, maxlen=max_cap_len, padding="post")

    # Make dataset from data lists
    dataset = tf.data.Dataset.from_tensor_slices((feature_data, caption_data))

    # Shuffle and batch
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Return the made dataset
    return dataset, max_cap_len


def compute_axiom_frequency(proof_data_file, ids_file):
    with open(proof_data_file, "rb") as f:
        proof_data = pickle.load(f)

    ids = load_ids(ids_file)

    # Extract all the relevant axioms
    axioms = []
    for prob_id, data in proof_data.items():
        if "version" in data:
            prob_id = data["version"]

        # Skip problems not in the ID set
        if prob_id in ids:
            axioms.extend(data["axioms"])

    # Count the occurence of the axioms and return the result
    counts = Counter(axioms)
    return counts


def main():

    # Set order for testing purposes
    # order = None
    # order = 'lexicographic'
    order = AxiomOrder.FREQUENCY

    # Function for testing the dataset creation
    tokenizer_path = os.path.join(os.path.dirname(config.train_id_file), "tokenizer.json")
    tokenizer, _ = get_tokenizer(tokenizer_path)

    # Test with validation file as we want this to run quicker
    if order is AxiomOrder.FREQUENCY:
        axiom_frequency = compute_axiom_frequency(config.proof_data, config.val_id_file)
    else:
        axiom_frequency = None

    train_data, max_len_train = get_dataset(
        config.val_id_file,
        config.proof_data,
        config.problem_features,
        tokenizer=tokenizer,
        order=order,
        axiom_frequency=axiom_frequency,
    )

    test_data, max_len_test = get_dataset(
        config.test_id_file,
        config.proof_data,
        config.problem_features,
        tokenizer=tokenizer,
        max_cap_len=max_len_train,
        order=order,
        axiom_frequency=axiom_frequency,
        remove_unknown=True,
    )

    print(train_data)
    print(test_data)
    print("Max length: ", max_len_train)
    assert max_len_test == max_len_train


if __name__ == "__main__":
    main()
