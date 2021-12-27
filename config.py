import sys

PYTHON = sys.executable


# TODO are these needed?
PROVERS = ["iprover", "e", "vampire"]
COMPETITIONS = ["hl4", "jjt"]

COMPETITION_RESULTS = {
    "jjt": {
        "e": "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/E---LTB-2.6/",
        "vampire": "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/Vampire---LTB-4.6/",
        "iprover": "http://www.tptp.org/CASC/28/WWWFiles/Results/JJT/iProver---LTB-3.5/",
    },
    "hl4": {
        "e": "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/E---LTB-2.5/",
        "iprover": "http://www.tptp.org/CASC/J10/WWWFiles/Results/HL4/iProver---LTB-3.3/",
    },
}


# #############
DEVELOPING = True
#DEVELOPING = False

BATCH_SIZE = 32
BUFFER_SIZE = 1000
EPOCHS = 1
ES_PATIENCE = 10


train_id_file = "data/vampire_jjt_fof/train.txt"
test_id_file = "data/vampire_jjt_fof/test.txt"
val_id_file = "data/vampire_jjt_fof/val.txt"

base_model = "experiments/base_model"

# TODO rename these variables as it is v confusing
proof_data = "data/raw/vampire_jjt.pkl"
problem_features = "data/embeddings/jjt_fof_sine_1_0.pkl"

TOKEN_PAD = "<pad>"
TOKEN_START = "<start>"
TOKEN_END = "<end>"
TOKEN_OOV = "<unk>"
TOKEN_DELIMITER = "\n"
