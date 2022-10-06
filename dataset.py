import pickle
from collections import Counter
import tensorflow as tf
import random
import os
import json
import glob
from multiprocessing import Pool
from keras.preprocessing.text import tokenizer_from_json

# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
from utils import debug
from enum_types import AxiomOrder, EncoderInput
from pathlib import Path
import numpy as np
from subprocess import check_call

import config


def get_tokenizer(id_file, tokenizer_mode, tokenizer_data_path, vocab_word_limit):

    tokenizer_path = get_tokenizer_save_path(
        os.path.dirname(id_file),
        Path(id_file).stem,
        tokenizer_mode,
        vocab_word_limit,
    )

    # Check if the set tokenzier does not exist
    if not os.path.exists(tokenizer_path):
        print(f"Cannot find tokenizer for: {tokenizer_path}")
        print("Computing new tokenizer given the parameters ...")

        # Make cmd string for computing the tokenizer and run the script
        cmd = f"{config.PYTHON} compute_tokenizer.py --id_file {id_file} "
        cmd += f" --tokenizer_data_path {tokenizer_data_path} --tokenizer_mode {tokenizer_mode} "
        if vocab_word_limit is not None:
            cmd += f" --vocab_word_limit {vocab_word_limit} "

        check_call(cmd, shell=True, stdout=None)

    # Load and return (tokenizer,  vocab limit)
    return load_tokenizer(tokenizer_path)


def get_caption_conjecture_tokenizers(model_params, proof_data, context, train_id_file, problem_features):

    # Get pre-trained tokenizer for the captions - supply context for switching between axioms and natural language
    caption_tokenizer, vocab_size = get_tokenizer(
        train_id_file, str(context), proof_data, model_params.target_vocab_size
    )

    if model_params.encoder_input is EncoderInput.SEQUENCE:
        conjecture_tokenizer, conjecture_vocab_size = get_tokenizer(
            train_id_file, "conj_word", problem_features, model_params.input_vocab_size
        )
        # Need to set the conjecture vocab size if it was None|all, to determine the input layer
        if model_params.input_vocab_size is None:
            model_params.input_vocab_size = conjecture_vocab_size
    else:
        conjecture_tokenizer = None

    return caption_tokenizer, vocab_size, conjecture_tokenizer


def get_tokenizer_save_path(dest, id_file, tokenizer_mode, vocab_word_limit) -> str:

    # Convert None to all for better name representation
    if vocab_word_limit is None:
        vocab_word_limit = "all"

    # Save the tokenizer
    save_path = os.path.join(dest, f"tokenizer_{tokenizer_mode}_{id_file}_{vocab_word_limit}.json")

    return save_path


def load_tokenizer(tokenizer_path, verbose=1):
    # Function for loading the tokenizer from path
    with open(tokenizer_path) as f:
        tokenizer = tokenizer_from_json(json.load(f))

    # Get the vocab size - might not already be set
    vocab_size = tokenizer.num_words
    if vocab_size is None:
        vocab_size = len(tokenizer.word_index)

    if verbose > 0:
        print(f"Loaded tokenizer from path {tokenizer_path}")
        print(f"Vocabulary Size: {vocab_size}")

    return tokenizer, vocab_size


def load_doc(filename):
    # Return the text contained in a document
    with open(os.path.expanduser(filename), "r") as file:
        text = file.read()
    return text


def max_caption_length(captions):

    # Check whether caption is tokenized or a single string with delimiters
    first_cap = list(captions.values())[0]
    if isinstance(first_cap, list):
        max_len = max(len(s) for s in list(captions.values()))
    elif isinstance(first_cap, str):
        max_len = max(len(s.split(config.TOKEN_DELIMITER)) for s in list(captions.values()))
    else:
        raise ValueError(
            f"Check type of the provided captions. Unclear how to compute max length of caption like: {first_cap}"
        )

    return max_len


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

def create_axiom_descriptions(axioms, order, axiom_frequency):

    # Replace delimiter with spaces
    axioms = [ax.replace(config.TOKEN_DELIMITER, " ") for ax in axioms]

    # Order the axioms if set
    if order is not None:
        axioms = order_axioms(order, axioms, axiom_frequency=axiom_frequency)

    # Build the caption string and return
    return (f"{config.TOKEN_START}{config.TOKEN_DELIMITER}"
            + f"{config.TOKEN_DELIMITER}".join(axioms)
            + f"{config.TOKEN_DELIMITER}{config.TOKEN_END}")

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

            # Build a string caption description of the axiom
            descriptions[prob_id] = create_axiom_descriptions(axioms, order, axiom_frequency)

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


def load_conjecture_tokens_dict(conjecture_path, conjecture_tokenizer, ids, conjecture_input_length):

    # Load the features from the pickle file
    conjectures = load_clean_conjectures(conjecture_path, ids)

    for prob_id, conj in conjectures.items():
        conjectures[prob_id] = conjecture_tokenizer.texts_to_sequences([conj])[0]

    for prob_id, conj in conjectures.items():
        conjectures[prob_id] = pad_sequences(
            [conj], maxlen=conjecture_input_length, padding="post", truncating="post"
        )[0]

    return conjectures


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


def tokenize_description_dicts(captions, caption_tokenizer, remove_unknown):

    # Tokenize each caption and store it back in the dictionary
    if remove_unknown:
        caption_tokenizer.oov_token = None  # This skips entries that maps to oov

    for i in captions:
        captions[i] = caption_tokenizer.texts_to_sequences([captions[i]])[0]

    return captions


def load_caption_dict(
    captions_path, ids, order, axiom_frequency, caption_tokenizer, max_cap_len, remove_unknown
):
    """
    Load the captions as a dictionary and tokenize if tokenizer is provided.
    Also compute the length of the longest caption if not provided.
    """
    captions = load_clean_descriptions(captions_path, ids, order=order, axiom_frequency=axiom_frequency)

    # Tokenize the sentences if provided
    if caption_tokenizer is not None:
        captions = tokenize_description_dicts(captions, caption_tokenizer, remove_unknown)

    # Compute the longest caption if value not provided - do this after tokenization in case remove_unknown is set
    if max_cap_len is None:
        max_cap_len = max_caption_length(captions)

    return captions, max_cap_len


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


def load_entity_features(encoder_input, entity_feature_path, ids, conjecture_tokenizer, conjecture_input_length):

    # Load the encoder input
    if encoder_input is EncoderInput.FLAT:
        # Load image features as a dict and get flag of whether they are cached
        entity_features, caching = load_image_feature_dict(entity_feature_path, ids)
    elif encoder_input is EncoderInput.SEQUENCE:
        entity_features = load_conjecture_tokens_dict(
            entity_feature_path, conjecture_tokenizer, ids, conjecture_input_length
        )
        # We always set caching to False for sequence input for now
        caching = False
    else:
        raise ValueError(f"Unrecognised EncoderInput type for loading input data: {encoder_input}")

    return entity_features, caching


def load_warmstart_problem_description(problem_path, caption_tokenizer, order, remove_unknown, axiom_frequency):

    # Open the file
    with open(problem_path, "r") as f:
        data = f.readlines()

        # Remove the conjecture
        prob_axioms = [d.strip() for d in data if 'conjecture' not in d]

    # Transform axioms into descriptions
    caption = create_axiom_descriptions(prob_axioms, order, axiom_frequency)[:-1] # Remove end token

    # Tokenize the captions
    if remove_unknown:
        caption_tokenizer.oov_token = None  # This skips entries that maps to oov
    caption = caption_tokenizer.texts_to_sequences([caption])[0]

    return Path(problem_path).name, caption


def load_warmstart_data(ids, dirpath, caption_tokenizer, order, remove_unknown, axiom_frequency, workers=None):

    # Compute all problem paths
    problem_paths = [os.path.join(dirpath, i) for i in ids]
    star_args = [(prob, caption_tokenizer, order, remove_unknown, axiom_frequency) for prob in problem_paths]

    pool = Pool(workers)
    res = pool.starmap(load_warmstart_problem_description, star_args)
    pool.close()
    pool.join()

    # Extract results
    captions = dict(res)

    return captions


def get_dataset(
    ids_path,
    captions_path,
    entity_feature_path,
    caption_tokenizer=None,
    max_cap_len=None,
    batch_size=config.BATCH_SIZE,
    order=None,
    axiom_frequency=None,
    remove_unknown=False,
    encoder_input=EncoderInput.FLAT,
    conjecture_tokenizer=None,
    conjecture_input_length=None,
):

    # Load the necessary data for the id set
    ids = load_ids(ids_path)

    # Load the entity features
    entity_features, caching = load_entity_features(encoder_input, entity_feature_path, ids, conjecture_tokenizer, conjecture_input_length)

    # Load the captions as a dict
    captions, max_cap_len = load_caption_dict(
        captions_path, ids, order, axiom_frequency, caption_tokenizer, max_cap_len, remove_unknown
    )

    # Build data lists for creating a tf.dataset
    caption_data = []
    feature_data = []
    for i in ids:
        # Remove captions that have reduced to start+end tokens due to unknown removal
        if not remove_unknown or (remove_unknown and len(captions[i]) > 2):
            caption_data.append(captions[i])
            feature_data.append(entity_features[i])
    # Delete dict variables to save memory
    del captions
    del entity_features

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
    tokenizer, _ = load_tokenizer(tokenizer_path)

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
    """

    load_conjecture_tokens_dict(
        "data/raw/deepmath_conjectures.pkl",
        "data/deepmath/tokenizer_conjecture_None.json",
        load_ids("data/deepmath/train.txt"),
        200,
    )


if __name__ == "__main__":
    main()
