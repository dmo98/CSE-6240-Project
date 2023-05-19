# CSE 6240 Spring 2023 Group Project

Richard Rex Arockiasamy, Michael Tang, Daanish M Mohammed, Neng Kai Nigel Neo

# Abstract
In this paper, we explore the ability of Graph Classification Models to learn to classify graphs based on the graph structure alone without relying on node or edge features. The dataset we will be using to tacke this problem is the REDDIT-BINARY dataset, which contains a set of graphs, where each graph represents discussions between users on a Reddit post. The objective is to exploit the underlying graph structure to classify each graph/post as belonging to one of two kinds of subreddits: question/answer-based subreddits or discussion-based subreddits, which makes this a Binary Classification problem on graphs. We will be using a few Graph Neural Network (GNN) models to learn to distinguish the posts into the 2 categories based on the graph structure. We also plan on applying an attention-based model, known as the Graph Attention Network (GAT), to this problem.

# Running the code
## Prerequisites
1. PyTorch Geometric
1. PyTorch
1. scikit-learn
1. matplotlib
1. numpy

## Steps to run
The dataset is available from PyTorch Geometric and running `train_model.py` will automatically download the dataset first before training and evaluating the model. 

```
python train_model.py -h
```
