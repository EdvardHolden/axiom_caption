import pickle
from collections import Counter
import tensorflow as tf
import random
import os
import json
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from utils import debug, AxiomOrder
import numpy as np

import config


def get_tokenizer(tokenizer_path, verbose=0):
    # Function for loading the tokenizer from path
    with open(tokenizer_path) as f:
        tokenizer = tokenizer_from_json(json.load(f))

    # Get the vocab size - might not already be set
    vocab_size = tokenizer.num_words
    if vocab_size is None:
        vocab_size = len(tokenizer.word_index)

    if verbose > 0:
        print("Vocabulary Size: %d" % vocab_size)
    return tokenizer, vocab_size


def load_doc(filename):
    # Return the text contained in a document
    with open(os.path.expanduser(filename), "r") as file:
        text = file.read()
    return text


def max_caption_length(captions):
    return max(len(s.split(config.TOKEN_DELIMITER)) for s in list(captions.values()))


def compute_max_caption_length(image_ids, image_descriptions):
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


def order_axioms(order, axioms, axiom_frequency=None):

    # We are expecting and order but want to chedck that the type is correct
    # and order != AxiomOrder.original:
    if not isinstance(order, AxiomOrder):
        raise ValueError(f"Incorrect value given for order: '{order}' needs to be of type AxiomOrder")
    if order is AxiomOrder.ORIGINAL:
        pass  # Do not change the order
    elif order is AxiomOrder.LEXICOGRAPHIC:
        axioms = sorted(axioms)
    elif order is AxiomOrder.LENGTH:
        axioms = sorted(axioms, key=len)
    elif order is AxiomOrder.RANDOM:
        # Randomly mutates the list
        random.shuffle(axioms)
    elif order is AxiomOrder.FREQUENCY or AxiomOrder.RANDOM_GLOBAL:
        if axiom_frequency is None:
            raise ValueError(f"Order is set to {order}, but no frequency list is supplied")
        axioms = sorted(axioms, key=lambda x: -axiom_frequency[x])
    else:
        raise ValueError(f"No ordering function implemented for order: '{order}'")

    return axioms


def load_clean_conjectures(filename, ids):
    # Load the descriptions of the images in the set and append start/end tokens
    with open(os.path.expanduser(filename), "rb") as f:
        conjecture_data = pickle.load(f)

    conjectures = dict()
    for prob_id, data in conjecture_data.items():

        # Skip problems not in the ID set
        if prob_id in ids:
            conjectures[prob_id] = data

    return conjectures


def load_clean_descriptions(filename, ids, order, axiom_frequency=None):
    # Load the descriptions of the images in the set and append start/end tokens
    with open(os.path.expanduser(filename), "rb") as f:
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

            # Order the axioms if set
            if order is not None:
                axioms = order_axioms(order, axioms, axiom_frequency=axiom_frequency)

            # Build the caption string and save in dict
            descriptions[prob_id] = (
                f"{config.TOKEN_START}{config.TOKEN_DELIMITER}"
                + f"{config.TOKEN_DELIMITER}".join(axioms)
                + f"{config.TOKEN_DELIMITER}{config.TOKEN_END}"
            )

    return descriptions


def load_photo_features(filename, dataset):
    # Load the img features of the photos in the set
    with open(os.path.expanduser(filename), "rb") as f:
        all_features = pickle.load(f)

    # Ensure no new line added to the name entry
    for k in sorted(all_features):
        all_features[k.strip()] = all_features.pop(k)

    # FIXME Adding the .p extension in the name for now - should be made more uniform
    if ".p" == str(all_features[list(all_features.keys())[0]][-2:]):
        features = {k: all_features[k + ".p"] for k in dataset}
    else:
        features = {k: all_features[k] for k in dataset}

    return features


def _load_cached_features(img_name, cap):
    img_tensor = np.load(img_name.decode("utf-8") + ".npy")
    return img_tensor, cap


def load_image_feature_dict(image_feature_path, ids):
    """
    Load the image features as a pre-step for making tf.dataset.
    If a directory si provided it caches the *.npy files in that
    directory instead of loading all the features matrixes.
    """

    caching = False  # Whether we are using cached image features
    if os.path.isfile(os.path.expanduser(image_feature_path)):
        # Load the features from the pickle file
        img_features = load_photo_features(image_feature_path, ids)
    else:
        # Build feature dictionary with a path to the cached data
        caching = True
        img_features = {
            prob_id: os.path.expanduser(os.path.join(image_feature_path, prob_id)) for prob_id in ids
        }

    return img_features, caching


def load_caption_dict(
    captions_path, ids, order, axiom_frequency, caption_tokenizer, max_cap_len, remove_unknown
):
    """
    Load the captions as a dictionary and tokeenize if tokenizer is provided
    """
    captions = load_clean_descriptions(captions_path, ids, order=order, axiom_frequency=axiom_frequency)

    # Compute the longest caption if value not provided
    if max_cap_len is None:
        max_cap_len = max_caption_length(captions)

    # Tokenize the sentences if provided
    if caption_tokenizer is not None:
        # Tokenize each caption and store it back in the dictionary
        if remove_unknown:
            caption_tokenizer.oov_token = None  # This skips entries that maps to oov
        for i in captions:
            captions[i] = caption_tokenizer.texts_to_sequences([captions[i]])[0]

    return captions


def create_tf_dataset(feature_data, caption_data, caching, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((feature_data, caption_data))

    if caching:
        # Add function which loads the cached features
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                _load_cached_features, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Shuffle and batch
    dataset = dataset.shuffle(config.BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataset(
    image_ids,
    captions_path,
    image_feature_path,
    caption_tokenizer=None,
    max_cap_len=None,
    batch_size=config.BATCH_SIZE,
    order=None,
    axiom_frequency=None,
    remove_unknown=False,
):

    # Load the necessary data for the id set
    ids = load_ids(image_ids)

    # TODO need to add in the sequences here somehow..
    # TODO maybe just drop the caching if we are using the other mode?? WHERE to place it????
    # TODO if we use conjectures we also need to alter the model into a sequence encoder and not jsut an image encoder
    # TODO need to load tokenized conejctures with this as well

    # Load image features as a dict and get flag of whether they are cached
    img_features, caching = load_image_feature_dict(image_feature_path, ids)

    # Load the captions as a dict
    captions = load_caption_dict(
        captions_path, ids, order, axiom_frequency, caption_tokenizer, max_cap_len, remove_unknown
    )

    # Build data lists for creating a tf.dataset
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
    if caption_tokenizer:
        caption_data = pad_sequences(caption_data, maxlen=max_cap_len, padding="post")

    # Make dataset from data lists
    print("Dataset size: ", len(feature_data))
    dataset = create_tf_dataset(feature_data, caption_data, caching, batch_size)

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


def compute_random_global_axiom_frequency(proof_data_file):
    """
    To compute a random global order on the axioms, we assign random values
    to the set of axioms in the vocabulary.
    """

    with open(proof_data_file, "rb") as f:
        proof_data = pickle.load(f)

    # Extract all the relevant axioms
    axioms = set()
    for _, data in proof_data.items():

        # Update set of axioms
        axioms.update(data["axioms"])

    # Get a list of number and shuffle them
    numbers = list(range(len(axioms)))
    random.shuffle(numbers)

    # Assign the numbers randomly to the axioms
    counts = {ax: num for ax, num in zip(axioms, numbers)}

    return counts


def main():

    # Set order for testing purposes
    # order = None
    # order = 'lexicographic'
    order = AxiomOrder.FREQUENCY

    # Function for testing the dataset creation
    tokenizer_path = os.path.join(os.path.dirname(config.train_id_file), "tokenizer.json")
    tokenizer, _ = get_tokenizer(tokenizer_path)

    """
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

    deepmath_data, max_len_deepmath = get_dataset(
        "data/deepmath/all.txt",
        config.proof_data,
        "data/embeddings/deepmath/deepmath_embedding_unsupervised_conjecture.pkl",
        tokenizer=tokenizer,
        max_cap_len=max_len_train,
        order=order,
        axiom_frequency=axiom_frequency,
        remove_unknown=True,
    )
    """

    print("### Test feature caching")
    deepmath_data, max_len_deepmath = get_dataset(
        "~/caption_attention/data/train.txt",
        "~/caption_attention/data/captions_6000_1.pkl",
        "~/caption_attention/data/image_features_6000_1",
        tokenizer=tokenizer,
        max_cap_len=50,
        order=AxiomOrder.ORIGINAL,
        axiom_frequency=None,
        remove_unknown=False,
    )


if __name__ == "__main__":
    main()
