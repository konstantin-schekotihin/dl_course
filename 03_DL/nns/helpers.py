import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


def plot_binary(predictor, X, y):
    plt.figure(figsize=(8,8))
    plt.scatter(X[:,0], X[:,1], c='w')
    
    # plot decision boundaries
    ax = plt.gca()
    y1, y2 = ax.get_ylim()
    x1, x2 = ax.get_xlim()
    xm, ym = np.meshgrid(np.arange(x1,x2,(x2-x1)/100), np.arange(y1,y2,(y2-y1)/100))
    p = predictor(np.c_[xm.ravel(), ym.ravel()]).reshape(xm.shape)
    plt.scatter(xm, ym, c=p)
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='w', s=100, linewidths=2)
    plt.show()

def rescale(X, min=-1, max=1):
    X_std = (X - X.min()) / (X.max() - X.min())
    return X_std * (max - min) + min    

def fn_and(X):
    return torch.logical_and(X[:,0], X[:,1]).float()

def fn_or(X):
    return torch.logical_or(X[:,0], X[:,1]).float()

def fn_xor(X): 
    return torch.logical_xor(X[:,0], X[:,1]).float()

def switch_fn(fn):
    return {'and': fn_and, 'or' : fn_or, 'xor' : fn_xor}.get(fn)

def annotate(im, ax):
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            ax.text(j, i, round(im[i, j].item()*100)/100, ha="center", va="center", color="r", fontsize=14, weight='bold')

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.figure(figsize=(15, 5))
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.show()

def plot_graph(G, classes):
    G = to_networkx(G, to_undirected=True)
    nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=False,
                     node_color=classes)
    plt.axis('off')
    plt.show()

def plot_embedding(t, classes, epoch=None, loss=None):
    t = t.detach().cpu().numpy()
    plt.scatter(t[:, 0], t[:, 1], s=140, c=classes)
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()