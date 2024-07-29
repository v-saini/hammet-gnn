
import sys
import urllib.request
from collections import defaultdict
#import pickle
import dgl
import numpy as np
import pandas as pd
import networkx as nx
from math import sqrt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Draw import IPythonConsole
from tqdm import tqdm
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch_geometric.utils import to_dgl

import os
import os.path as osp
import re
from typing import Callable, Dict, Optional, Tuple, Union

from torch_geometric.data import InMemoryDataset
from sklearn.metrics import r2_score

from dgllife.model import WeavePredictor, AttentiveFPPredictor, GCNPredictor, GATPredictor, NFPredictor

from utils import *
import argparse

setup_seed(32)
# Parse command line arguments
parser = argparse.ArgumentParser(description='Cross validating different Torch models')
parser.add_argument('model_name', type=str, help='Name of the model to cross validate (afp, afp_final, gcn, weave, gat, nf)')
args = parser.parse_args()


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
df_g['y'] = y
df_g['smiles'] = df['SMILES']

device = torch.device('cpu')

# Define model loading based on model_name argument
model_name = args.model_name
if model_name == 'afp_final':
    exp_configure = {"in_node_feats": 53, "in_edge_feats": 10, "num_layers": 5, "num_timesteps": 2,
                    "graph_feat_size": 192, "dropout": 0.2, "n_tasks": 1,
                    'lr': 0.0009695610504005811, 'weight_decay': 1.690436650288651e-05 }
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
    "graph_feat_size": 200,"dropout": 0}
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
    "num_gnn_layers": 2, "residual": True, "batchnorm": False}
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
    "predictor_hidden_feats": 32, "weight_decay": 1e-3}
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
    "predictor_hidden_feats": 128,"num_gnn_layers": 5, "residual": True}
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
    "num_gnn_layers": 5, "gnn_hidden_feats": 50,
    "graph_feats": 128, "gaussian_expand": True}
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


loss_func = torch.nn.MSELoss(reduce=None)
optimizer = torch.optim.Adam(model.parameters(), lr=exp_configure['lr'],  weight_decay=exp_configure['weight_decay'])

def train():
  reset_parameters(model)
  epochs = 100
  for epoch in range(1, epochs+1):
    model.train()
    train_loss = []
    for batch in train_dataloader:

      # Do a forward pass
      batch_graph, target = batch
      batch_graph = batch_graph.to(device)
      target = target.to(device)
      node_feats = batch_graph.ndata["x"]
      edge_feats = batch_graph.edata["edge_attr"]
      if model_name in ['afp', 'afp_final', 'weave']:
          predictions = model(batch_graph, node_feats, edge_feats)
      else:
          predictions = model(batch_graph, node_feats)
      loss = (loss_func(predictions, target.unsqueeze(1))).mean().sqrt()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss.append(loss)

def eval(X):
  predicted_values = []
  true_values = X.iloc[:, 1].tolist()
  model.eval()
  for graph_sample in X.iloc[:,0].tolist():
    graph_sample = graph_sample.to(device)
    node_feats = graph_sample.ndata["x"]
    edge_feats = graph_sample.edata["edge_attr"]
    if model_name in ['afp', 'afp_final', 'weave']:
        predictions = model(graph_sample, node_feats, edge_feats)
    else:
        predictions = model(graph_sample, node_feats)
    predicted_values.append(predictions.detach().item())
  r2 = r2_score(true_values,predicted_values)
  return r2

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=10)
train_r2 = []
test_r2 = []

  
for i, (train_index, test_index) in enumerate(kf.split(df_g)):
  
  Train = pd.DataFrame(df_g.iloc[train_index].reset_index(drop=True))
  Test = df_g.iloc[test_index].reset_index(drop=True)
  train_dataloader = torch.utils.data.DataLoader(dataset=list(zip(Train[0].tolist(), torch.tensor(Train['y'].tolist(), dtype=torch.float32))),
                                                batch_size=64, collate_fn=collate_data, shuffle=True)
  test_dataloader = torch.utils.data.DataLoader(dataset=list(zip(Test[0].tolist(), torch.tensor(Test['y'].tolist(), dtype=torch.float32))),
                                                batch_size=64, collate_fn=collate_data, shuffle=True)

  train()
  train_pred = eval(Train)
  test_pred = eval(Test)
  print(f"For {i}th fold Train R2: {train_pred}")
  print(f"For {i}th fold Test R2: {test_pred}")
  train_r2.append(train_pred)
  test_r2.append(test_pred)
print(f"The 5-fold train cross validation score is {sum(train_r2)/5}")
print(f"The 5-fold test cross validation score is {sum(test_r2)/5}")

  