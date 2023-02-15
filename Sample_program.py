import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch
import torch_geometric.data as data
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, train_test_split_edges, to_dense_adj
from torch_geometric.loader import DataLoader
from sklearn import preprocessing
from torch_geometric.nn import Node2Vec
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import argparse


device = "cpu"


parser = argparse.ArgumentParser(description="Introduce program parameters.")
parser.add_argument('-ROOT_DIR',nargs='?',const="./Coles",default="./Coles", type=str,action="store",help="Root directory for files")
parser.add_argument('-TEST_SIZE',nargs='?',const=0.2,default=0.2, type=float,action="store",help="Test size for training")
parser.add_argument('-TOLERANCE',nargs='?',const=1e-1,default=1e-1,type=float,action="store",help = "Tolerance for the embedding")
parser.add_argument('-N_SIM',nargs='?',const=2, type=int,default=2,action="store",help = "Number of simulations")
parser.add_argument('-TREATMENT',nargs='?',const=0,default=0,type=int,action="store",help="Treatment to apply")
parser.add_argument("-VERBOSE", nargs='?', const=0, default=0, type=int, action="store", help="Verbose (0/1)")
args = parser.parse_args()

ROOT_DIR = args.ROOT_DIR
TEST_SIZE = args.TEST_SIZE
TOLERANCE = args.TOLERANCE
N_SIM = args.N_SIM
TREATMENT = args.TREATMENT
VERBOSE = args.VERBOSE
print('-'*20)
print('PARAMETERS')
print('-'*20)
for key,value in vars(args).items():
    print(f"{key} -->  {value} ")
print('-'*20)


#ROOT_DIR = "./Coles"
#TEST_SIZE = 0.2
#TOLERANCE = 1e-0
#N_SIM = 3

def import_data(root_dir):
    """Files should be included in a directory /Nodes
        with a directory of Nodes in .csv format and
        another one of Edges in .csv
    """
    for root, dirs, files in os.walk(root_dir):
        if "Nodes" in root:
            node_files = {file: pd.read_csv(os.path.join(root, file)) for file in files}
        if "Edges" in root:
            edge_files = {file: pd.read_csv(os.path.join(root, file)) for file in files}

    for edge_file in edge_files.keys():
        node_file = edge_file.replace("Edges","Nodes")
        edge_files[edge_file]["escuela"] = int(''.join(c for c in edge_file if c.isdigit()))

        df_nodes = node_files[node_file]
        df_edges = edge_files[edge_file]
        if df_edges["to"].nunique() > df_edges["from"].nunique():
            new_vect = np.arange(df_edges["to"].nunique())
            ###
            translation = dict(zip(list(df_edges.sort_values(by="to")["to"].unique()), new_vect))
        else:
            new_vect = np.arange(df_edges["from"].nunique())
            ###
            translation = dict(zip(list(df_edges.sort_values(by="from")["from"].unique()), new_vect))
        node_files[node_file].replace({"ID": translation}, inplace=True)
        edge_files[edge_file].replace({"from": translation, "to": translation, "weight": {1: 1, 2: 1, -1: 0, -2: 0}},
                                      inplace=True)
    return node_files, edge_files


def apply_treatment(treatment=0, *, node_files, edge_files):
    """
    :param treatment:
        Receive the .csvs and select the classification
        treatment = 0 implies random split
        treatment = 1 implies split according to group
    :param node_files: .csv including node information
    :param edge_files: .csv including edge information
    :return: arranged .csv
    """
    for file in edge_files:
        if treatment == 0:
            class_classification = ["Missing"] * edge_files[file].shape[0]
        if treatment == 1:
            class_classification = []
            for edge in edge_files[file].iterrows():
                try:
                    node_file = "Nodes_"+file[file.index("t"):file.index(".")]+'.csv'
                    agent_from = node_files[node_file].iloc[list(node_files[node_file]["ID"]).index(edge[1]["from"])]
                    agent_to = node_files[node_file].iloc[list(node_files[node_file]["ID"]).index(edge[1]["to"])]
                    if (agent_from["Curso"] == agent_to["Curso"]):
                        class_classification.append(str(agent_from["Curso"]))
                    else:
                        class_classification.append("Intergroup")
                except:
                    class_classification.append("Missing")

        edge_files[file]["Classification"] = class_classification
    return


#def create_test_split(node_files, edge_files, test_size):
#    train_set = {}
#    test_set = {}
#    for file in edge_files:
#        train_edges, test_edges = train_test_split(edge_files[file], test_size=test_size)
#        train_set[file] = train_edges
#        test_set[file] = test_edges
#    return train_set, test_set

def create_test_split(embedding, test_size,treatment,course_test = -1):
    """
    :param treatment: The different treatment applied in the paper
    :param embedding: Input embedding obtained from the Node2Vec algorithm
    :param test_size: Size of the test set
    :return: Train and test embeddings
    """
    if treatment == 0:
        train_embedding, test_embedding = train_test_split(embedding, test_size=test_size)
    if treatment == 1 :
        target_course = course_test
        test_embedding = embedding[embedding["classification"] == course_test]
        train_embedding = embedding[embedding["classification"] != course_test]
    return train_embedding, test_embedding

def extract_embedding(edge_file, tolerance):
    """
    :param dataloader: Pytorch dataloader object with edge information for a single dataset
    :param edge_files: Edge files in order to include additional information for a single dataset
    :return: embedding
    """
    dl_object = data.Data(edge_index=torch.tensor(edge_file[["from", "to"]].to_numpy().T))

    model = Node2Vec(dl_object.edge_index, embedding_dim=128, walk_length=30,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=4, sparse=True).to(device)
    loader = model.loader(batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    #####
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    pre_value_loss, curr_value_loss = 100, 0
    epoch = 0
    while (abs(pre_value_loss - curr_value_loss) > tolerance):
        loss = train()
        epoch += 1
        pre_value_loss = curr_value_loss
        curr_value_loss = loss
        # if epoch % 5 == 0:
        #    print(f'Epoch: {epoch:02d}, with loss: {loss:.4f}')
    # print(f'The Node2vec algorithm converged at epoch: {epoch:02d}, with loss: {loss:.4f}')

    z = model()
    # from tensor to numpy
    emb_128 = z.detach().cpu().numpy()

    edge_embedding = []
    for u, v in dl_object.edge_index.t():
        edge_embedding.append(np.maximum(emb_128[u], emb_128[v]))
    embeddings = pd.DataFrame(edge_embedding)
    embeddings["escuela"] = edge_file["escuela"].values
    embeddings["weight"] = edge_file["weight"].values
    embeddings["classification"] = edge_file["Classification"].values

    return embeddings


def balanced_accuracy(y_true, y_pred):
    """Returns 0.5*sum_{for each class}(TPR)"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    clases = list(set(y_true))
    # return sum([sum((y_true == y_pred)&(y_true == clase))/np.count_nonzero(y_true==clase) for clase in clases])/len(clases)
    # non - pythonic, do it clearly
    tpr_total = 0
    for clase in clases:
        tpr = sum((y_true == y_pred) & (y_true == clase)) / np.count_nonzero(y_true == clase)
        tpr_total += tpr

    return tpr_total / len(clases)


def neural_network_training(train_embedding, test_embedding):
    X_train = train_embedding.drop(["escuela", "weight", "classification"], axis=1).values
    y_train = train_embedding["weight"].values
    X_test = test_embedding.drop(["escuela", "weight", "classification"], axis=1).values
    y_test = test_embedding["weight"].values

    ros = SMOTE(random_state=0, sampling_strategy="minority")
    emb_x_resampled, emb_y_resampled = ros.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(max_depth=7, class_weight="balanced")
    clf.fit(emb_x_resampled, emb_y_resampled)

    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        save_best_only=True)
    #######
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(train_embedding.shape[1]-3,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["AUC"])

    model_history = model.fit(emb_x_resampled, emb_y_resampled, epochs=500, verbose=0, batch_size=128,
                              validation_data=(X_test, y_test),
                              callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    return balanced_accuracy(y_test, clf.predict(X_test)),balanced_accuracy(y_test,np.round(model.predict(X_test,verbose=0)))


import time

prev_time = time.perf_counter()
# Prepare data
nodes, edges = import_data("./Coles")
apply_treatment(TREATMENT,node_files=nodes, edge_files=edges)
#train_set, test_set = create_test_split(nodes, edges, test_size=0.2)
print(f"Data Prepared ! Duration: {time.perf_counter() - prev_time:.2f} seconds")
print('-'*20)


# Embedding generation
prev_time = time.perf_counter()
total_embeddings = {}

for file in edges:
    total_embeddings[file] = extract_embedding(edges[file], tolerance=TOLERANCE)
print(f"Embeddings Prepared ! Duration: {time.perf_counter() - prev_time:.2f} seconds")
print('-'*20)
total_accuracy_RF = []
total_accuracy_NN = []
embedding_keys = list(total_embeddings.keys())

_ = 0
prev_time = time.perf_counter()
while _ < N_SIM:
    curr_key = embedding_keys[np.mod(_,N_SIM)]
    if TREATMENT == 1:
        courses_test = [item for item in total_embeddings[curr_key]["classification"].unique() if item.isdigit()]
        for course in courses_test:
            train_embedding,test_embedding= create_test_split(total_embeddings[curr_key], test_size=TEST_SIZE,treatment=TREATMENT,course_test=course)

            accuracy_RF, accuracy_NN = neural_network_training(train_embedding, test_embedding)

            total_accuracy_RF.append(accuracy_RF)
            total_accuracy_NN.append(accuracy_NN)
            _ += 1

            if VERBOSE == 1:
                print(f"Performed on {course=} of school {curr_key}\n "
                      f"Random Forest accuracy:{accuracy_RF:.2f} \n "
                      f"Neural network accuracy: {accuracy_NN:.2f}.")
                print('-' * 20)
                print(
                    f"{_ / N_SIM * 100:.2f} % completado, Time : {time.perf_counter() - prev_time :.2f} seconds since start of simulations")
                print('-' * 20)

            if _ == N_SIM:
                break







    elif TREATMENT == 0 :
        train_embedding, test_embedding = create_test_split(total_embeddings[curr_key], test_size=TEST_SIZE,
                                                        treatment=TREATMENT)
        accuracy_RF,accuracy_NN =neural_network_training(train_embedding, test_embedding)

        total_accuracy_RF.append(accuracy_RF)
        total_accuracy_NN.append(accuracy_NN)
        _ += 1

        if VERBOSE == 1:
            print(f"Performed randomly on school {curr_key}\n "
                  f"Random Forest accuracy:{accuracy_RF:.2f} \n "
                  f"Neural network accuracy: {accuracy_NN:.2f}.")
            print('-' * 20)
            print(
                f"{_ / N_SIM * 100:.2f} % completado, Time : {time.perf_counter() - prev_time :.2f} seconds since start of simulations")
            print('-' * 20)






with open(ROOT_DIR +'/accuracy_RF.txt', 'w') as fp:
    fp.write('\n'.join([str(item) for item in total_accuracy_RF]))

with open(ROOT_DIR +'/accuracy_NN.txt', 'w') as fp2:
    fp2.write('\n'.join([str(item) for item in total_accuracy_NN]))

print("-"*20)
print(f"The average accuracy for the Random Forest method is {np.mean(total_accuracy_RF):.2f} with and std of {np.std(total_accuracy_RF):.2f}")
print(f"The average accuracy for the Neural Network method is {np.mean(total_accuracy_NN):.2f} with and std of {np.std(total_accuracy_NN):.2f}")
