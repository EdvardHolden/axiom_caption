import pickle
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

# Need to supply file
parser = argparse.ArgumentParser()
parser.add_argument("feature_dict", help="Path to dictionary containing the feature entries to transform")


def main():

    # Get the feature dict
    feature_dict_path = parser.parse_args().feature_dict

    # Check that it is a pkl file
    if not Path(feature_dict_path).suffix == ".pkl":
        raise ValueError("Provided file must be a .pkl file")

    # Load pickle
    with open(feature_dict_path, "rb") as f:
        data = pickle.load(f)
    print(f"Number of features acquired: {len(data)}")

    # Create new directory
    dest = feature_dict_path[:-4]
    if not os.path.exists(dest):
        os.mkdir(dest)
    print(f"Writing result to: {dest}")

    # Process into npy files
    for feat_id, feat_array in tqdm(data.items()):
        np.save(os.path.join(dest, feat_id), feat_array)

    # Alert if the number of files in the dest dir is not the same as the original
    assert len(data) == len(
        os.listdir(dest)
    ), "Warning: Destination contains files not from current file computation"


if __name__ == "__main__":
    main()
