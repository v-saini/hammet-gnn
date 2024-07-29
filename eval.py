import sys
import urllib.request
from collections import defaultdict
import pickle
import dgl
import numpy as np
import pandas as pd
import networkx as nx
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
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate different Torch models')
parser.add_argument('model_name', type=str, help='Name of the model to evaluate (afp, afp_final, gcn, weave, gat, nf)')
args = parser.parse_args()

# Load your dataset and split into train and test
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

hammet = MyOwnDataset(root='data/')

g = [to_dgl(data) for data in hammet]

df_g = pd.DataFrame(g)

with open('train_index', 'rb') as fp:
    train_index = pickle.load(fp)

trainX = df_g[df_g.index.isin(train_index)]
testX = df_g[~df_g.index.isin(train_index)]

trainy = y[y.index.isin(train_index)]
testy = y[~y.index.isin(train_index)]

train_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(trainX.iloc[:, 0].tolist(),
                     torch.tensor(trainy.tolist(), dtype=torch.float32))),
    batch_size=64, collate_fn=collate_data, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    dataset=list(zip(testX.iloc[:, 0].tolist(),
                     torch.tensor(testy.tolist(), dtype=torch.float32))),
    batch_size=64, collate_fn=collate_data)

# Define model loading based on model_name argument
model_name = args.model_name
if model_name == 'afp_final':
    model = torch.load('final_models/afp_final.pth')
elif model_name == 'afp':
    model = torch.load('final_models/afp_default.pth')
elif model_name == 'gcn':
    model = torch.load('final_models/gcn_default.pth')
elif model_name == 'nf':
    model = torch.load('final_models/nf_default.pth')
elif model_name == 'gat':
    model = torch.load('final_models/gat_default.pth')
elif model_name == 'weave':
    model = torch.load('final_models/weave_default.pth')
else:
    raise ValueError(f"Model '{model_name}' not recognized.")

loss_func = torch.nn.MSELoss(reduction='mean')
device = torch.device('cpu')

def eval(loader, model):
    model.eval()
    valid_loss = []
    valid_targets = []
    valid_predictions = []
    with torch.no_grad():
        for batch in loader:
            batch_graph, target = batch
            batch_graph = batch_graph.to(device)
            target = target.to(device)
            node_feats = batch_graph.ndata["x"]
            edge_feats = batch_graph.edata["edge_attr"]
            if model_name in ['afp', 'afp_final', 'weave']:
                predictions = model(batch_graph, node_feats, edge_feats)
            else:
                predictions = model(batch_graph, node_feats)
            loss = torch.sqrt(loss_func(predictions, target.unsqueeze(1)))
            valid_loss.append(loss.item())
            valid_targets.extend(target.cpu().numpy())
            valid_predictions.extend(predictions.cpu().numpy())

    # Compute R^2 for validation set
    rmse = np.mean(valid_loss)
    valid_r2 = r2_score(valid_targets, valid_predictions)
    print(f'Predictions:')
    print('--------------------------------')
    return valid_r2, rmse

train_pred = eval(train_dataloader, model)
test_pred = eval(test_dataloader, model)

print(f"Train results: R2 = {train_pred[0]}, RMSE = {train_pred[1]}")
print(f"Test results: R2 = {test_pred[0]}, RMSE = {test_pred[1]}")
