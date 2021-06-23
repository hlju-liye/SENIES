# SENIES
 SENIES is a deep learning based two-layer predictor for enhancing the identification of enhancers and their strength by utilizing DNA shape information beyond two common sequence-derived features, namely kmer and one-hot.

# Requirements
* python = 2.7
* numpy = 1.16.0
* torch = 1.4.0
* scikit-learn = 0.20.4

# Usage
All the data used for training our models has been prepared in this repository. All you have to do is cloning it. Then you can simply change into the code subdirectory and run the following command to start training for the first and second layers.

* `python senies.py -l 1` or `./senies.py -l 1`
* `python senies.py -l 2` or `./senies.py -l 2`





