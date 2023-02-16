# Triadic influence as a proxy for compatibility in social relationships

This repository contains the scripts that reproduce the results presented in our paper:

Triadic influence as a proxy for compatibility in social relationships, Ruíz-García<sup>*</sup>, M., Ozaita<sup>*</sup>, J., Pereda, M., Alfonso, A., Branas-Garza, P., Cuesta, J. A., and Sánchez, A., Proceedings of the National Academy of Sciences, XXX

the data collected in this work is hosted here:

https://zenodo.org/record/7647000#.Y-5eDtLMJH4

it contains the social network (weighted and directed) of the relationships in 13 middle schools in Spain, containing 3395 students and 60566 relationships.


The paper contains two parts corresponding to different parts of the paper: a first one corresponding to the importance of triadic influence to link prediction in social network analysis; and a second one where we use embedding theory. 


.....


For the second part, we include a sample program (sample_program.py) to represent the process done during the paper, where accuracy measures for the different treatments can be obtained given different school datasets. This program should be executed with the data from Zenodo displayed in a single folder named "Data" with two subfolders named Nodes and Edges with the mentioned files in it. 

The program is meant to be executed in the console where the following parameters may be included: 

ROOT_DIR  --------  Indicate the root directory of the Data folder
TEST_SIZE --------  Indicate the test size for the first treatment of the paper 
TOLERANCE --------  Indicate the tolerance for the fitting of the embedding
N_SIM     --------  Indicate the number of simulations in order to create the statistics
TREATMENT --------  Indicate the treatment to be used
VERBOSE   --------  Indicate the verbose of the process, if 1 then the additional information besides accuracy is included. 
