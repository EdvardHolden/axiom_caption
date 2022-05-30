import matplotlib.pyplot as plt
import pickle
import os

from parser import get_plot_result_parser


def main():

    # Parse input arguments
    parser = get_plot_result_parser()
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
