# IMPORTS
print('Importing libraries')
# from functools import  partial
from datetime import datetime
import json
import argparse
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import numpy as np
import itertools

from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import roc_auc_score

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]  = (20,3)

print("Torch version: {}".format(torch.__version__))
print("Torch Geometric version: {}".format(torch_geometric.__version__))

print('Imports completed!')


def download_data(data_dir):
    # download the dataset, split into train/test
    pyg_dataset = TUDataset(root=data_dir, name="REDDIT-BINARY", use_node_attr=True)
    pos, neg = pyg_dataset[:1000], pyg_dataset[1000:]
    pos, neg = pos.shuffle(), neg.shuffle()
    return pos, neg

# GNN model class for GraphSAGE & GAT
class GraphModel(nn.Module):
    def __init__(self, in_feats, num_graph_layers, hidden_graph_feats, hidden_linear_feats, out_feats, layer_type, aggr='add'):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == 'gat':
            self.layers = [GATConv(in_feats, hidden_graph_feats, concat=False)]
            self.layers.extend([GATConv(hidden_graph_feats, hidden_graph_feats, concat=False) for _ in range(1, num_graph_layers)])
        elif layer_type == 'gcn':
            self.layers = [GCNConv(in_feats, hidden_graph_feats)]
            self.layers.extend([GCNConv(hidden_graph_feats, hidden_graph_feats) for _ in range(1, num_graph_layers)])
        elif layer_type == "sage":
            self.layers = [SAGEConv(in_feats, hidden_graph_feats, aggr=aggr)]
            self.layers.extend([SAGEConv(hidden_graph_feats, hidden_graph_feats, aggr=aggr) for _ in range(1, num_graph_layers)])
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        self.linear1 = nn.Linear(hidden_graph_feats, hidden_linear_feats)
        self.linear2 = nn.Linear(hidden_linear_feats, out_feats)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = self.act(layer(x, edge_index))

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_graph_feats]

        # Apply a final classifier
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)

        return x
    

def calculate_metrics(model, dataloader: torch.utils.data.DataLoader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    y_true = []
    y_pred = []
    for data in dataloader:  # Iterate in batches over the test dataset.
        x = torch.ones(data.num_nodes).unsqueeze(dim=1)
        out = model(x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float().unsqueeze(dim=1))
        y_true += data.y.tolist()
        y_pred += out.squeeze().tolist()
        pred = (out > 0.5).int().squeeze()  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        total_loss += loss.item()

    accuracy = correct / len(dataloader.dataset)
    aucroc_score = roc_auc_score(y_true, y_pred)
    return accuracy, total_loss / len(dataloader.dataset), aucroc_score


# Helper function for train/eval
def train_eval(config, train_dataset, test_dataset): 
    print('Started executing train/eval loop')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # initialize the GNN model
    model = GraphModel(in_feats=1, 
                        num_graph_layers=config['graph_layers'], 
                        hidden_graph_feats=config['graph_feats'], 
                        hidden_linear_feats=config['linear_feats'], 
                        out_feats=1, 
                        layer_type=config['layer_type'],
                        aggr=config['agg'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCELoss(reduction="sum")

    train_losses = []
    train_accuracies = []
    train_aucroc_scores = []

    test_losses = []
    test_accuracies = []
    test_aucroc_scores = []

    for epoch in range(1, config['epochs'] + 1):
        # run training loop
        model.train()

        #batch_loss = []
        train_loss_over_batch = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.

            x = torch.ones(data.num_nodes).unsqueeze(dim=1)
            out = model(x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y.float().unsqueeze(dim=1))  # Compute the loss.
            train_loss_over_batch += loss.item()

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

        # evaluate on train
        train_acc, train_loss, train_aucroc_score = calculate_metrics(model, train_loader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_aucroc_scores.append(train_aucroc_score)

        # evaluate on test
        test_acc, test_loss_over_batch, test_aucroc_score = calculate_metrics(model, test_loader, criterion)
        test_losses.append(test_loss_over_batch)
        test_accuracies.append(test_acc)
        test_aucroc_scores.append(test_aucroc_score)

        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Test loss: {test_loss_over_batch:.4f}, Test AUCROC Score : {test_aucroc_score:.4f}')

        # tune.report(loss=test_loss, accuracy=test_acc)
    return train_losses, train_accuracies, train_aucroc_scores, test_losses, test_accuracies, test_aucroc_score


def draw_epoch_plot(config, train_metric, test_metric, metric_name, show_error=True):
    title = f"{config['layer_type'].upper()} {metric_name} vs. epochs"
    fig_size = (10, 5)
    fig = plt.figure(figsize=fig_size)

    xaxis = np.arange(1, config['epochs'] + 1)
    # plot means
    plt.plot(xaxis, np.mean(train_metric, axis=0), "b-", label="train")
    plt.plot(xaxis, np.mean(test_metric, axis=0), "r-", label="test")
    # plot error bars
    if show_error is True:
        plt.errorbar(xaxis, np.mean(train_metric, axis=0), yerr=np.std(train_metric, axis=0), fmt="b,", capsize=5)
        plt.errorbar(xaxis, np.mean(test_metric, axis=0), yerr=np.std(test_metric, axis=0), fmt="r,", capsize=5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    return fig


def cv(config, pos, neg, plot=False):
    folds = config['k']
    dataset_size = len(pos)
    fold_size = dataset_size // folds

    train_losses = np.zeros((folds, config['epochs']))
    train_accuracies = np.zeros((folds, config['epochs']))
    train_aucs = np.zeros((folds, config['epochs']))
    test_losses = np.zeros((folds, config['epochs']))
    test_accuracies = np.zeros((folds, config['epochs']))
    aucs = np.zeros((folds, config['epochs']))
    for k in range(folds):

        def get_fold(dataset, k):
            if k == 0:
                test_dataset = dataset[:fold_size]
                train_dataset = dataset[fold_size:]
            elif k == folds-1:
                train_dataset = dataset[:fold_size * (folds-1)]
                test_dataset = dataset[fold_size * (folds-1):]
            else:
                train_dataset = dataset[:fold_size * k] + dataset[fold_size * (k+1):]
                test_dataset = dataset[fold_size * k : fold_size * (k+1)]
            return train_dataset, test_dataset
        
        pos_train, pos_test = get_fold(pos, k)
        neg_train, neg_test = get_fold(neg, k)
        train_dataset = pos_train + neg_train
        test_dataset = pos_test + neg_test
        
        train_losses[k, :], train_accuracies[k, :], train_aucs[k, :], \
        test_losses[k, :], test_accuracies[k, :], aucs[k, :] = train_eval(config, train_dataset, test_dataset)


    print(json.dumps(config, indent=2))

    test_loss = np.mean(test_losses[:, -1])
    test_accuracy = np.mean(test_accuracies[:, -1])
    auc = np.mean(aucs[:, -1])

    print(f'Mean Test loss after CV : {test_loss}+-{np.std(test_losses[:, -1])}')
    print(f'Mean Test Accuracy after CV: {test_accuracy}+-{np.std(test_accuracies[:, -1])}')
    print(f"AUCROC across CVs: {auc}+-{np.std(aucs[:, -1])}")
    
    # plot graphs
    if plot:
        nowtime = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        fig = draw_epoch_plot(config, train_metric=train_losses, test_metric=test_losses, metric_name="Loss")
        fig.savefig(f"figures/loss-{nowtime}.pdf")
        fig = draw_epoch_plot(config, train_metric=train_accuracies, test_metric=test_accuracies, metric_name="Accuracy", show_error=False)
        fig.savefig(f"figures/accuracy-{nowtime}.pdf")
        fig = draw_epoch_plot(config, train_metric=train_aucs, test_metric=aucs, metric_name="AUCROC", show_error=False)
        fig.savefig(f"figures/aucroc-{nowtime}.pdf")
        print("Saved figures.")

    return test_loss, test_accuracy, auc
    

def config_to_string(config):
    hashable = ""
    for key in config:
        hashable += f"{key}={config[key]}_"
    return hashable[:-1]

def main(args):
    torch_geometric.seed_everything(1)

    # download the dataset, split into train/test
    pos, neg = download_data(data_dir="./pyg_dataset")
    print('Downloaded dataset')

    # define the model training configuration

    default_config = vars(args)

    if args.tune:

        best_config = None
        exp_config = default_config.copy()

        results = {}

        options = {
            'lr': [0.001, 0.005, 0.01],
            'graph_layers': [2, 3, 4],
            'graph_feats': [16, 32, 64],
        }

        for params in itertools.product(*list(options.values())):

            for index, key in enumerate(options.keys()):
                exp_config[key] = params[index]
            
            exp_config['linear_feats'] = exp_config['graph_feats']
            print("Running Config: \n")
            print(json.dumps(exp_config, indent=2))

            test_loss, test_acc, auc = cv(exp_config, pos, neg, plot=False)

            temp_config = config_to_string(exp_config)
            results[temp_config] = [test_loss, test_acc, auc]

            if test_acc > results.get(best_config, [0, 0, 0])[1]:
                best_config = temp_config

        print(f"Best config was: {best_config}")
        print(json.dumps(results, indent=2))

        with open("results_loss_acc_auc.json", "w") as outfile:
            json.dump(results, outfile)

    else:
        print(json.dumps(default_config, indent=2))
        cv(default_config, pos, neg, plot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help='number of cross-validation folds')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs for train model for')
    parser.add_argument('--graph_layers', type=int, default=3, help='number of layers in the GNN model')
    parser.add_argument('--graph_feats', type=int, default=32, help='size of hidden layer embeddings in the GNN')
    parser.add_argument('--linear_feats', type=int, default=32, help='size of hidden layer dimension from linear layers')
    parser.add_argument('--layer_type', type=str, default='gcn', help='type of layer in GNN', choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--agg', type=str, default='mean', help='aggregation method to use')
    parser.add_argument('--tune', type=bool, default=False, help='option to perform hyperparameter search')

    args = parser.parse_args()
    main(args)