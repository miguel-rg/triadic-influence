# Triadic influence as a proxy for compatibility in social relationships

This repository contains the scripts that reproduce the results presented in our paper:

Triadic influence as a proxy for compatibility in social relationships, Ruíz-García<sup>*</sup>, M., Ozaita<sup>*</sup>, J., Pereda, M., Alfonso, A., Branas-Garza, P., Cuesta, J. A., and Sánchez, A., Proceedings of the National Academy of Sciences, XXX

the data collected in this work is hosted here:

https://zenodo.org/record/7647000#.Y-5eDtLMJH4

it contains the social network (weighted and directed) of the relationships in 13 middle schools in Spain, containing 3395 students and 60566 relationships.

This repository contains two code files corresponding to the two main parts of the paper: the definition and use of triadic influence and the use of node embeddings to train neural networks.

## Triadic influence and statistical analysis of the data

To facilitate reproducibility we have uploaded a Google Colab notebook that should run on the cloud:

https://github.com/miguel-rg/triadic-influence/blob/main/triadic_influence_data_analysis_and_training_NN.ipynb

This notebook automatically downloads the necessary data, and produces several statistical analysis of it. It also computes the triadic influence between students and uses it to train neural networks to predict the sign of the relationships.

## Training on node embeddings 

We also include a sample program (sample_program.py) that can reproduce the results presented in the paper for the case of training neural networks with node embeddings.

This program should be executed with the data from Zenodo displayed in a single folder named "Data" with two subfolders named Nodes and Edges with the mentioned files in it. 

More information about the creation of node embeddings can be found at **XXX (should we mention node2vec?)**

The program can be executed from the console where the following parameters may be included: 

ROOT_DIR  --------  Indicate the root directory of the Data folder

TEST_SIZE --------  Indicate the test size

TOLERANCE --------  Indicate the tolerance for the fitting of the embedding

N_SIM     --------  Indicate the number of simulations in order to create the statistics

TREATMENT --------  Indicate the treatment to be used

VERBOSE   --------  Indicate the verbose of the process, if 1 then the additional information besides accuracy is included. 
