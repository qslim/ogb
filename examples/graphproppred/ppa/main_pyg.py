import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def main():
    dataset = DglGraphPropPredDataset(name = 'ogbg-ppa')
    
    for data in dataset:
        print(data)

if __name__ == "__main__":
    main()