import sys

PYTHON = sys.executable

# #############
# DEVELOPING = True
DEVELOPING = False

BATCH_SIZE = 64
# BATCH_SIZE = 256
# BUFFER_SIZE = 5000
BUFFER_SIZE = 1000
#EPOCHS = 80
EPOCHS = 2
ES_PATIENCE = 5
#ES_PATIENCE = None


train_id_file = "data/deepmath/train.txt"
# train_id_file = "data/deepmath/val.txt"
# train_id_file = "data/deepmath/test.txt"
test_id_file = "data/deepmath/test.txt"
val_id_file = "data/deepmath/val.txt"

base_model = "experiments/base_model"

# TODO rename these variables as it is v confusing
proof_data = "data/raw/deepmath.pkl"
problem_features = "data/embeddings/deepmath/graph_features_deepmath_all.pkl"
#problem_features = "data/embeddings/deepmath/graph_features_deepmath_premise.pkl"

TOKEN_PAD = "<pad>"
TOKEN_START = "<start>"
TOKEN_END = "<end>"
TOKEN_OOV = "<unk>"
TOKEN_DELIMITER = "\n"

CONJECTURE_INPUT_MAX_LENGTH = 500
