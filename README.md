# Axiom Caption - Captioning conjectures with axioms

Project for premise selection in first-order theorem proving by drawing parallels to the task of image captioning.
The conjecture (+surrounding axioms) are encoded into a graph and embedded as features. These onjecture vectors
are given to an ML model, which along with a \<start\> token start generating axioms. 

# Getting Started - Data Preparation

### Downloading the proof data

1. First download the axioms used in a proof from the CASC competition by using `python3 download_axioms.py` (change the settings as appropriate).
2. Next, find the problem versions which was used in each proof in CASC by running `python3 download_casc_mety.py`. This results in a meta-dictionarr which contains a mapping between the problem names and version of each prover.
3. (Optional) pre-process the problems with sine using `sine_process_problems.py`

### Creating datasets

Run `create_train_test_split.py` over a given axiom dict. This will split the ids into train/test/val sets.

# Computing conjecture embeddings

TODO


## Testing
No tests so far.

## Authors

* **Edvard Holden** 

