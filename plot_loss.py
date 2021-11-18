import matplotlib.pyplot as plt
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")


def main():

    # Parse input arguments
    args = parser.parse_args()

    # Load training history
    with open(os.path.join(args.model_dir, 'history.pkl'), 'rb') as f:
        hist = pickle.load(f)

    plt.plot(hist["train_loss"])
    plt.plot(hist["val_loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend(['train', 'val'], loc='upper right')

    plt.show()
    plt.close('all')


if __name__ == "__main__":
    main()
