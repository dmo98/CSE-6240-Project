# IMPORTS
print('Importing libraries')
from functools import  partial
import argparse
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.nn import GATConv, GCNConv, SSGConv
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import roc_auc_score

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from train_model import download_data, train_eval
from train_model import GraphModel

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]  = (20,3)

print("Torch version: {}".format(torch.__version__))
print("Torch Geometric version: {}".format(torch_geometric.__version__))

print('Imports completed!')



def cv(config, pos, neg):
    folds = config['k']
    dataset_size = len(pos)
    fold_size = dataset_size // folds

    test_losses = torch.zeros((folds,))
    test_accuracies = torch.zeros((folds,))
    test_aucs = []
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
        
        _, _, _, test_loss, test_accuracy, test_aucroc = train_eval(config, train_dataset, test_dataset)
        test_losses[k] = test_loss
        test_accuracies[k] = test_accuracy
        test_aucs.append(test_aucroc)
    
    print(f'Mean Test loss after CV : {torch.mean(test_losses)}+-{torch.std(test_losses)}')
    print(f'Mean Test Accuracy after CV: {torch.mean(test_accuracies)}+-{torch.std(test_accuracies)}')
    print(f"AUCROC across CVs: {np.mean(test_aucs)}+-{np.std(test_aucs)}")

    tune.report(loss=torch.mean(test_losses), accuracy=torch.mean(test_accuracies), aucroc=torch.mean(test_aucs))



def main(args):
    # download the dataset
    pos, neg = download_data(data_dir="./pyg_dataset", split=args.split)
    print('Downloaded dataset')

    # define the hyperparameter search space
    config = {
        'k': args.k,
        'batch_size': tune.choice([16, 32, 64]),
        'epochs': 10,
        'lr': tune.loguniform(7e-4, 3e-2),
        'graph_layers': tune.choice([1, 2, 3, 4, 5]),
        'graph_feats': tune.choice([16, 24, 32, 40, 48, 56, 64]),
        'linear_feats': tune.choice([16, 24, 32, 40, 48, 56, 64]),
        'layer_type': args.layer_type,
    }

    # to terminate bad trials early
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    
    # report the performance of hp on test set 
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    

    # hp_tuner = tune.Tuner(
    #     partial(train_eval, train_dataset=train_dataset, test_dataset=test_dataset),
    #     param_space=config,
    # )
    # results = hp_tuner.fit()
    hp_tuner = tune.run(
        partial(cv, pos=pos, neg=neg),
        config = config,
        num_samples = args.samples,
        progress_reporter = reporter,
        scheduler=scheduler,
        )
    
    best_model = hp_tuner.get_best_trial(metric='loss', mode='min')
    print("Best model's hyperparameter configuration: {}".format(best_model.config))
    print("Best model's validation loss: {}".format(best_model.last_result['loss']))
    print("Best model's validation accuracy: {}".format(best_model.last_result['accuracy']))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10, help='number of random hyperparameter samples')
    parser.add_argument('--split', type=float, default=0.8, help='train/test split ratio')
    parser.add_argument('--k', type=int, default=5, help='number of fold in cross validation')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    # parser.add_argument('--epochs', type=int, default=50, help='max epochs for train model for')
    # parser.add_argument('--in_feats', type=int, default=0, help='size of each input sample')
    # parser.add_argument('--num_graph_layers', type=int, default=10, help='number of layers in the GNN model')
    # parser.add_argument('--hidden_graph_feats', type=int, default=10, help='size of hidden layer embeddings in the GNN')
    # parser.add_argument('--hidden_linear_feats', type=int, default=10, help='size of hidden layer dimension from linear layers')
    # parser.add_argument('--out_feats', type=int, default=2, help='size of output of GNN')
    parser.add_argument('--layer_type', type=str, default='gcn', help='type of layer in GNN')
    parser.add_argument('--agg', type=str, default='mean', help='aggregation method to use')

    args = parser.parse_args()
    main(args)
