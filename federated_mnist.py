import glob
import os
import random
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from federated import FederatedManager, consume_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def seed(n=0):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)


def load_dsets():
    return (
        datasets.MNIST(
            'data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        datasets.MNIST(
            'data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
    )


def index_to_dataset(y, p=None):
    classes = set(y.numpy())
    n_classes = len(classes)
    p = p or 1/n_classes
    pnot = (1-p)/(n_classes-1)
    ps = np.full((n_classes, n_classes), pnot)
    np.fill_diagonal(ps, p)
    return np.array([np.random.choice(10, p=ps[yi]) for yi in y])


def make_fed_dloaders(dset, p=None, batch_size=64, shuffle=True):
    
    return [DataLoader(dset, batch_size, shuffle) for dset in make_fed_dsets(dset, p=p)]
    
def make_fed_dsets(dset, p=None):
    _, y = consume_dataset(dset)
    classes = set(y.numpy())
    n_classes = len(classes)
    idx_dset = index_to_dataset(y, p=p)
    dset_idx = [np.where(idx_dset == di)[0] for di in range(n_classes)]
    return [Subset(dset, di) for di in dset_idx]


def check_dsets(dsets):
    for dset in dsets:
        _, y = consume_dataset(dset)
        print(Counter(y.numpy()))


def plot(manager, filename="out/loss.pdf", fig=None, label=None):
    if fig is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
    else:
        ax = fig.get_axes()
    ax[0].plot(manager.history["test_loss"], label=label)
    ax[1].plot(manager.history["test_acc"], label=label)
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Round")
    if label:
        ax[1].legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    return fig, ax


def train(p=None, batch_size=64, n_rounds=100, lr=1e-2, momentum=0.5,
          one_dset=False, outdir="out/", runname="latest", n_training=None):
    os.makedirs(outdir, exist_ok=True)
    train_dset, test_dset = load_dsets()
    if n_training:
        train_dset = Subset(train_dset, np.arange(n_training))
    stacked_dsets = (
        make_stacked_dsets(train_dset, p=p) if not one_dset else [train_dset]
    )
    check_dsets(stacked_dsets)
    stacked_dloaders = [DataLoader(dset, batch_size=batch_size, shuffle=True)
                        for dset in stacked_dsets]

    manager = FederatedManager(
        stacked_dloaders,
        mnist.Net,
        nn.NLLLoss(),
        test_dset,
        lr=lr,
        momentum=momentum
    )

    for i in range(n_rounds):
        print("Round", i)
        manager.round()
        print(
            "Global",
            manager.history["test_loss"][-1],
            manager.history["test_acc"][-1]
        )
        joblib.dump((runname, manager), filename=outdir+runname+".pkl")
        plot(manager, filename=outdir+runname+".pdf", label=runname)
        plt.close("all")

    return manager


def train_all():
    seed(n=42)
    train(one_dset=True, runname="benchmark")
    train(runname="uniform")
    train(p=0.9, runname="p09")
    train(p=1.0, runname="p10")


def load_managers(outdir="out/"):
    return [joblib.load(pklfile) for pklfile in glob.glob(outdir+"*.pkl")]


def plot_all(managers):
    fig = None
    for runname, manager in managers:
        fig, ax = plot(manager, fig=fig, label=runname)


def main():
    train_all()
    managers = load_managers()
    plot_all(managers)