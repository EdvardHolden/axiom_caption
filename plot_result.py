import matplotlib.pyplot as plt
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Directory containing the history.pkl of interest")
parser.add_argument(
    "--metric",
    default="loss",
    choices=["loss", "l", "coverage", "c", "jaccard", "j"],
    help="The metric to plot",
)


def main():

    # Parse input arguments
    args = parser.parse_args()

    # Quick hack for abbrevations
    metric = args.metric
    if metric == "l":
        metric = "loss"
    elif metric == "j":
        metric = "jaccard"
    elif metric == "c":
        metric = "coverage"

    # Load training history
    with open(os.path.join(args.model_dir, "history.pkl"), "rb") as f:
        hist = pickle.load(f)

    plt.plot(hist[f"train_{metric}"])
    plt.plot(hist[f"val_{metric}"])
    plt.xlabel("Epochs")
    plt.ylabel(f"{metric.capitalize()}")
    plt.title(f"{metric.capitalize()} Plot")
    plt.legend(["train", "val"], loc="upper right")

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
