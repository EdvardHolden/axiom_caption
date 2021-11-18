import config
import argparse
from nltk.translate.bleu_score import corpus_bleu
import os
import tensorflow as tf
from tqdm import tqdm

from dataset import get_dataset, get_tokenizer, compute_max_length
from model import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--problem_ids', default=config.test_id_file,
                    help='File containing IDs for evaluation')

parser.add_argument('--problem_features', default=config.problem_features,
                    help="File containing the image features")
parser.add_argument('--proof_data', default=config.proof_data,
                    help="File containing the image descriptions")
parser.add_argument('--axiom_order', default=None, choices=[None, 'default', 'length', 'lexicographic'],
                    help='The order of the axioms in caption')

parser.add_argument('--max_length', default=None, type=int,
                    help='The maximum length of the predictions')
parser.add_argument('-v', '--verbose', action='count', default=0)


def generate_step(tokenizer, model, max_len, img_tensor):

    # List for storing predicted sequence
    result = []

    # Reset LSTM states between each batch
    model.reset_states()

    # Get start token
    dec_input = tf.expand_dims([tokenizer.word_index[config.TOKEN_START]], 0)

    # Run the model until we reach the max length or the end token
    for i in range(max_len):
        # Predict probabilities
        predictions = model([img_tensor, dec_input])
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        # Return sequence if we predicted the end token
        if tokenizer.index_word[predicted_id] == config.TOKEN_END:
            return result

        # Add predicted ID to the result
        result.append(tokenizer.index_word[predicted_id])

        # Set predicted word as the next model input
        dec_input = tf.expand_dims([predicted_id], 0)

    # Reached max length, returning the sequence
    return result


def evaluate_model(tokenizer, model, test_data, max_len, verbose=0):

    # Initialise results
    actual, predicted = list(), list()

    # Predict for each problem in the data
    for (_, (img_tensor, caption)) in tqdm(enumerate(test_data)):
        # Generate caption based on the image tensor
        yhat = generate_step(tokenizer, model, max_len, img_tensor)

        # Extract the string value from the tensor, remove start/end tokens,
        # convert to utf-8 and make into array
        caption = str(caption.numpy()[0]).split()[1:-1]
        # Store the actual token
        actual.append([caption])
        predicted.append(yhat)

        if verbose:
            print('Actual:    %s' % ' '.join(caption))
            print('Predicted: %s' % ' '.join(yhat))

    # Calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    return bleu


def main():

    # Get the arguments
    args = parser.parse_args()

    # Get pre-trained tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.problem_ids), 'tokenizer.json')
    tokenizer, _ = get_tokenizer(tokenizer_path)

    # If maximum length is not provided, we compute it based on the training set in config
    if args.max_length is None:
        max_len = compute_max_length(config.train_id_file, args.proof_data)
    else:
        max_len = args.max_length
    print("Max caption length: ", max_len)

    # Get the test dataset with batch 1 as we need to treat each caption separately
    # Also, we want the raw text so not providing a tokenizer
    test_data, _ = get_dataset(args.problem_ids, args.proof_data,
                               args.problem_features, batch_size=1,
                               order=args.axiom_order)

    # Load model
    model_dir = os.path.join(args.model_dir, 'ckpt_dir')
    loaded_model = load_model(model_dir)

    # Run evaluation
    score = evaluate_model(tokenizer, loaded_model, test_data, max_len, verbose=args.verbose)
    print("Score: ", score)


if __name__ == "__main__":
    main()
