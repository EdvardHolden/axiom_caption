# Axiom Caption - Captioning conjectures with axioms

Project for premise selection in first-order theorem proving by drawing parallels to the task of image captioning.
The conjecture and surrounding axioms are encoded into a graph and embedded as features. These problem vectors
are given to a generative ML model. The model is fed a \<start\> token which starts the generation process. 
This terminates when the model predicts the \<end\> token or the maximum prediction length is reached.


The parallel to image captioning is shows depicted bellow:
<img src="figs/Axiom Captioning Parallel.png" class="center">

The following Figure overview the system:
<img src="figs/Axiom Captioning Overview.png" class="center">


# Installation 

Scripts were made with Python 3.7.5

``pip install -r requirements.txt``

# Getting Started - Data Preparation

The DeepMath data can be found in the repository used for computing the problem embeddings (link further below).

### Downloading the proof data (from CASC)

1. First download the proof data with axioms from the CASC competitions by running `python3 download_axioms.py` (This will download all the viable proof data specified in the config).
2. (Optional) pre-process the problems with sine using `sine_process_problems.py`

### Creating datasets and tokenizer

Run `create_train_test_split.py` over a given axiom dict. This will split the ids into train/test/val sets.
To compute the tokenizer over the training set you run `python3 compute_tokenizer.py` with the appropriate parameters.
Now, you are able to start training models over the dataset.


### Computing problem embeddings

The problem embeddings are computed using graph neural networks and the adjoining scripts are found [here](https://github.com/EdvardHolden/gnn-entailment-caption)_.

### Defaults

The default paths are set in `config.py`


# Model Parameters

The models are initialised according to the `params.json` file. It's directory is supplied as the `--model_dir` parameter.
The parameters are as follows:

* model_type: ["inject_decoder", "inject", "dense"] - Set mode type. Dense is a normal dense model and inject_decoder is a faster version of inject.
* axiom_order: ["original", "lexicographic", "length", "frequency", "random", "random_global"] - How to order the axioms in the captions.
* attention: ["none", "bahdanau", "flat", "loung_dot", "loung_concat"] - Which attention mechanism (if any) to use for the sequence decoder models.
* stateful: [true, false] - whether the RNNs should be stateful.
* normalize: [true, false] - whether to normalize the emebdding input features.
* batch_norm: [true, false] - whether to apply batch normalization.
* rnn_type: [gru , lstm] - which RNN architecture to use.
* no_rnn_units: [int] - number of RNN units in the RNN.
* embedding_size: [int] - number of embedding dimensions for the input tokens
* no_dense_units: [int] - number of units in the dense layers.
* dropout_rate: [float] - Rate for dropout in RNN and dense layers.
* learning_rate: [float] - learning rate for the optimizer.
* remove_unknown: [true, false] - If set to true, removes axioms mapped to unkown by the tokenizer.
* axiom_vocab_size: [int|"all"] - Size of the axiom vicabulary for the captions.
* encoder_input: ["sequence", "flat"] - Flat for flat feature vectors and sequence for sequential style input.
* conjecture_vocab_size: [int|"all"] - Vocabulary size of the input conjectures. Only used in encoder_input is set to sequence.



# Experiments

Overview of how to run the different experiments.
Most follow a similar style of running a hyper parameter search over sets of training and model parameters.

## Experiment: Find the optimal graph embedding

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

Most of the following experiments follow a similar style.


## Experiment: Axiom Order

The goal of this experiment is to examine the impact of different axiom order when training the decoder.
Run the script `search_hyperparams.py` on a `params.json` file with the appropriate parameters.
For instance:
`
nohup python3 -u search_hyperparams.py --problem_features data/raw/deepmath_conjectures.pkl --parameter_space experiments/vocab_6000_experiment_2A/params.json  --experiment_dir experiments/vocab_6000_experiment_2A >> nohup.out  2>&1 &
`

## Experiment: Decoding and sampling

The aim of this experiment is to evaluate different sampling methods.
The script `run_experiment_decoding.py` runs the experiment, and samplers can be added into the `SAMPLER` constant in the script.

## Experiment: Attention mechanisms

The aim of this experiment is to determine the performance of various attention mechanisms, also achieved through `search_hyperparams.py`.
e.g.
`
nohup python3 -u search_hyperparams.py --problem_features data/raw/deepmath_conjectures.pkl --parameter_space experiments/vocab_6000_experiment_2B/params.json  --experiment_dir experiments/vocab_6000_experiment_2B >> nohup.out  2>&1 &
`

## Generating problems

Problems can be generated in multiple settings using the `generate_problems.py script`.
Below is an example which combines captioning with the model found in "model_dir", with SInE with unbounded depth and a tolerance of 3.

`
python3 -u generate_problems.py --problem_dir ~/gnn-entailment-caption/merged_problems/ --mode caption_sine --sine_st 3 --sine_sd 0 --no_samples 4 --problem_format mptp --result_postfix_ merged_features_all --feature_path ~/gnn-entailment-caption/embeddings/graph_features_mizar_merged_deepmath_all.pkl --output_format original --model_dir experiments/vocab_6000_experiment_2B/sequence/
`

The output_format is original fof but can be changed to clausified.

## Authors

* **Edvard Holden** 

