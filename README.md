# Axiom Caption - Captioning conjectures with axioms

Project for premise selection in first-order theorem proving by drawing parallels to the task of image captioning.
The conjecture (+surrounding axioms) are encoded into a graph and embedded as features. These onjecture vectors
are given to an ML model, which along with a \<start\> token start generating axioms. 

# Getting Started - Data Preparation

### Downloading the proof data

1. First download the proof data with axioms from the CASC competitions by running `python3 download_axioms.py` (This will download all the viable proof data specified in the config).
2. (Optional) pre-process the problems with sine using `sine_process_problems.py`

### Creating datasets

Run `create_train_test_split.py` over a given axiom dict. This will split the ids into train/test/val sets.

## Computing conjecture embeddings

TODO


## Testing
No tests so far.

## Authors

* **Edvard Holden** 

