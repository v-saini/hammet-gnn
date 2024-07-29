import sys
import urllib.request
from collections import defaultdict
import random
import argparse
import pickle
import dgl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from math import sqrt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Draw import IPythonConsole
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch_geometric.utils import to_dgl


import os
import os.path as osp
import re
from typing import Callable, Dict, Optional, Tuple, Union

from torch_geometric.data import InMemoryDataset, download_url, extract_gz
from sklearn.metrics import r2_score

from dgllife.model import WeavePredictor, MPNNPredictor, GATPredictor, AttentiveFPPredictor, GCNPredictor, NFPredictor
from utils import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Build different Torch models')
parser.add_argument('model_name', type=str, help='Name of the model to build (afp, afp_final, gcn, weave, gat, nf)')
args = parser.parse_args()

# Load data
df = pd.read_csv('data.csv')
x_smiles = df['SMILES']
y = df['Sigma']

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data_new.csv'

    @property
    def processed_file_names(self):
        return 'data.dt'

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

hammet = MyOwnDataset(root = 'data/')

g = []
for i in range(len(hammet)):
  data = to_dgl(hammet[i])
  g.append(data)

df_g = pd.DataFrame(g)

with open ('train_index', 'rb') as fp:
    train_index = pickle.load(fp)

trainX = df_g[df_g.index.isin(train_index)]

testX = df_g[~df_g.index.isin(train_index)]

trainy =  y[y.index.isin(train_index)]

testy =  y[~y.index.isin(train_index)]

device = torch.device('cpu')
model_name = args.model_name
if model_name == 'afp_final':
    exp_configure = {"in_node_feats": 53, "in_edge_feats": 10, "num_layers": 5, "num_timesteps": 2,
                 "graph_feat_size": 192, "dropout": 0.2, "n_tasks": 1,
                  'lr': 0.0009695610504005811, 'weight_decay': 1.690436650288651e-05, "batch_size": 64}
    model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']).to(device)
    
elif model_name == 'afp':
    exp_configure = {"lr": 3e-4, "weight_decay": 0,"in_node_feats" : 53, "n_tasks" : 1, "in_edge_feats" : 10,
    "patience": 30, "num_layers": 2, "num_timesteps": 2,
    "graph_feat_size": 200,"dropout": 0, "batch_size": 64}
    model = AttentiveFPPredictor(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']).to(device)
    
elif model_name == 'gcn':
    exp_configure = {"lr": 2e-2, "weight_decay": 0, "in_node_feats" : 53, "n_tasks" : 1, "patience": 30,
    "dropout": 0.05, "gnn_hidden_feats": 256, "predictor_hidden_feats": 128,
    "num_gnn_layers": 2, "residual": True, "batchnorm": False, "batch_size": 64}
    model = GCNPredictor(
            in_feats=exp_configure['in_node_feats'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']).to(device)
    
elif model_name == 'nf':
    exp_configure = {"lr": 1e-2, "in_node_feats" : 53, "n_tasks" : 1,  "batchnorm": False, "dropout": 0.15,
    "gnn_hidden_feats": 32, "num_gnn_layers": 2, "patience": 30,
    "predictor_hidden_feats": 32, "weight_decay": 1e-3, "batch_size": 64}
    model = NFPredictor(
            in_feats=exp_configure['in_node_feats'],
            n_tasks=exp_configure['n_tasks'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_size=exp_configure['predictor_hidden_feats'],
            predictor_batchnorm=exp_configure['batchnorm'],
            predictor_dropout=exp_configure['dropout']
        ).to(device)
    
elif model_name == 'gat':
    exp_configure = {"lr": 3e-4, "weight_decay": 0,"in_node_feats" : 53, "n_tasks" : 1, "patience": 30,
    "dropout": 0.05, "gnn_hidden_feats": 64,"num_heads": 8,"alpha": 0.06,
    "predictor_hidden_feats": 128,"num_gnn_layers": 5, "residual": True, "batch_size": 64}
    model = GATPredictor(
          in_feats=exp_configure['in_node_feats'],
          hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
          num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
          feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
          attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
          alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
          residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
          predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
          predictor_dropout=exp_configure['dropout'],
          n_tasks=exp_configure['n_tasks']
      ).to(device)
    
elif model_name == 'weave':
    exp_configure = {"lr": 3e-4, "weight_decay": 0, "in_node_feats" : 53, "n_tasks" : 1, "in_edge_feats" : 10, "patience": 30,
    "num_gnn_layers": 5, "gnn_hidden_feats": 50, "graph_feats": 128, "gaussian_expand": True, "batch_size": 64}
    model = WeavePredictor(
          node_in_feats=exp_configure['in_node_feats'],
          edge_in_feats=exp_configure['in_edge_feats'],
          num_gnn_layers=exp_configure['num_gnn_layers'],
          gnn_hidden_feats=exp_configure['gnn_hidden_feats'],
          graph_feats=exp_configure['graph_feats'],
          gaussian_expand=exp_configure['gaussian_expand'],
          n_tasks=exp_configure['n_tasks']
      ).to(device)
    
else:
    raise ValueError(f"Model '{model_name}' not recognized.")

# Create train and test dataloaders
train_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(trainX.iloc[:,0].tolist(),
                     torch.tensor(trainy.tolist(), dtype=torch.float32))),
    batch_size=exp_configure['batch_size'], collate_fn=collate_data, shuffle=True)


test_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(testX.iloc[:,0].tolist(),
                    torch.tensor(testy.tolist(), dtype=torch.float32))),
    batch_size=exp_configure['batch_size'], collate_fn=collate_data)

train_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(trainX.iloc[:,0].tolist(),
                     torch.tensor(trainy.tolist(), dtype=torch.float32))),
    batch_size=64, collate_fn=collate_data, shuffle=True)

# train_dataloader.dataset[0]

test_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(testX.iloc[:,0].tolist(),
                    torch.tensor(testy.tolist(), dtype=torch.float32))),
    batch_size=64, collate_fn=collate_data)

loss_func = torch.nn.MSELoss(reduce=None)

optimizer = torch.optim.Adam(model.parameters(), lr=exp_configure["lr"],  weight_decay=exp_configure["weight_decay"])

epochs = 100
setup_seed(32)
# loop over epochs
# model.reset_parameters()
tl_final = []
vl_final = []
reset_parameters(model)
for epoch in tqdm(range(1, epochs+1)):
    # Training loop
    model.train()
    train_loss = []
    train_targets = []
    train_predictions = []
    for batch in train_dataloader:
        batch_graph, target = batch
        batch_graph = batch_graph.to(device)
        target = target.to(device)
        node_feats = batch_graph.ndata["x"]
        edge_feats = batch_graph.edata["edge_attr"]
        if model_name in ['afp', 'afp_default', 'weave']:
            predictions = model(batch_graph, node_feats, edge_feats)
        else:
            predictions = model(batch_graph, node_feats)
        loss = (loss_func(predictions, target.unsqueeze(1))).mean().sqrt()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_targets.extend(target.cpu().numpy())
        train_predictions.extend(predictions.detach().numpy())

    # Compute R^2 for training set
    train_r2 = r2_score(train_targets, train_predictions)

    # Validation loop
    model.eval()
    valid_loss = []
    valid_targets = []
    valid_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch_graph, target = batch
            batch_graph = batch_graph.to(device)
            target = target.to(device)
            node_feats = batch_graph.ndata["x"]
            edge_feats = batch_graph.edata["edge_attr"]
            if model_name in ['afp', 'afp_default', 'weave']:
                predictions = model(batch_graph, node_feats, edge_feats)
            else:
                predictions = model(batch_graph, node_feats)
            loss = (loss_func(predictions, target.unsqueeze(1))).mean().sqrt()
            valid_loss.append(loss.item())
            valid_targets.extend(target.cpu().numpy())
            valid_predictions.extend(predictions.detach().numpy())

    # Compute R^2 for validation set
    valid_r2 = r2_score(valid_targets, valid_predictions)

    # Print and store metrics
    if epoch%20 == 0:
        print('--------------------------------')
        print("Training loss:", np.mean(train_loss))
        print("Training R^2:", train_r2)
        print("Test loss:", np.mean(valid_loss))
        print("Test R^2:", valid_r2)
        tl_final.append(np.mean(train_loss))
        vl_final.append(np.mean(valid_loss))

def eval(X,y):
    predicted_values = []
    true_values = y.tolist()
    model.eval()
    for batch_graph in X.iloc[:,0].tolist():
        batch_graph = batch_graph.to(device)
        node_feats = batch_graph.ndata["x"]
        edge_feats = batch_graph.edata["edge_attr"]
        if model_name in ['afp', 'afp_default', 'weave']:
            predictions = model(batch_graph, node_feats, edge_feats)
        else:
            predictions = model(batch_graph, node_feats)
        predicted_values.append(predictions.detach().item())
    r2 = r2_score(true_values,predicted_values)
    #print("RMSE ", root_mean_squared_error(true_values,predicted_values))
    #   print(r2)
    pred = pd.concat([pd.DataFrame(true_values, columns=['true_values']), pd.DataFrame(predicted_values, columns=['predicted_values'])], axis=1)
    return r2

train_pred = eval(trainX, trainy)
test_pred = eval(testX, testy)
print(f"Train results: {train_pred}")
print(f"Test results: {test_pred}")


torch.save(model, 'final_models/weave_default.pth')
