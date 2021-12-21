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

# Experiments

Overview of how to run the different experiments.

## Experiment 1: Find the optimal graph embedding

The goal of this experiment is to find the best performing problem embedding from a set of embeddings.
The pre-requisite for this experiment is to compute different problem embeddings and place them in a folder (see above).

The experiment takes three parameters:
* model_config: The parameters for the base model used on all embedding configurations.
* embedding_dir: The directory containing the problem embeddings to use in this experiment.
* experiment_dir: The directory for storing the training results.


Run `python3 run_experiment_graph_embedding.py` to run the experiment. This will train the base
captioning model on the set dataset for each embedding configuration.
To inspecting the results run: `python3 synthesise_results.py` on the experiment_dir.
This will produce a 'results.md' containing the final model statistics.



## Testing
No tests so far.

## Authors

* **Edvard Holden** 

