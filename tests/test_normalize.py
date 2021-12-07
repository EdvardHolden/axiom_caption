import numpy as np
from keras.layers import Normalization
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append("/home/eholden/axiom_caption/")  # FIXME
from model import adapt_normalization_layer, get_model, get_model_params


def get_data():
    iris = load_iris()
    df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    df["target"] = pd.Series(iris["target"], name="target_values")
    df["target_name"] = df["target"].replace(
        [0, 1, 2], ["iris-" + species for species in iris["target_names"].tolist()]
    )
    X = df[df.columns[:3]].values
    return X


def get_scaled_data(X):
    return StandardScaler().fit_transform(X)


def get_normalized_data(X):
    layer = Normalization()
    layer.adapt(X)
    return np.array(layer(X))


# Maybe I also should check this for the actual model code?
# could check with output of the node
def main():
    X = get_data()

    x_scaled = get_scaled_data(X)
    x_norm = get_normalized_data(X)

    # Compute the number of data points
    no_entries = len(x_norm[0]) * len(x_norm)  # FIXME

    # Check that a large chunk of the normalised and scaled
    # data has actually changed fromt he original matrix
    assert sum(sum(np.isclose(x_scaled, X))) / no_entries < 0.94
    assert sum(sum(np.isclose(X, x_norm))) / no_entries < 0.94

    # Test that both transormations produces similar results
    assert sum(sum(np.isclose(x_scaled, x_norm))) / no_entries > 0.94

    # Check if model normalization output is close to the standard scaled data
    model_params = get_model_params("experiments/base_model")
    model = get_model("merge_inject", 24, 100, model_params)
    model = adapt_normalization_layer(model, X)
    x_model = model.image_encoder.normalize(X)
    assert sum(sum(np.isclose(x_scaled, x_model))) / no_entries > 0.94


if __name__ == "__main__":
    main()
