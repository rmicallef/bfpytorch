import torch
import torchvision


import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#from tqdm.notebook import trange
from collections import Counter
import numpy as np



data_path = './data'

default_training_batch_size = 64
default_testing_batch_size = 100
default_learning_rate = 0.01

import torch
import torch.nn as nn
import torch.nn.functional as F


# THIS IS THE OFFICIAL PYTORCH EXAMPLE, BUT IT'S SLOWER THAN THE OTHER MODEL, SO FOR NOW, IT'S FAST BEFORE TIDY
'''
class DefaultNet(nn.Module):
    def __init__(self):
        super(DefaultNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
'''

# Quick and gets the job done well enough for testing
class DefaultNet(nn.Module):
    def __init__(self):
        super(DefaultNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#'''

def get_MNIST_dataloaders():
    """
    Download the standard MNIST dataset if it isn't already here, and make torch DataLoaders using the 
    parameters from the torch example code.
    """
    
    # The normalization paramaters below are the mean and standard deviation of values in the MNIST set
    default_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

    training_dataset = torchvision.datasets.MNIST(root=data_path, download=True, train=True, transform=default_transform)
    testing_dataset = torchvision.datasets.MNIST(root=data_path, download=True, train=False, transform=default_transform)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=default_training_batch_size, shuffle=True)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=default_testing_batch_size, shuffle=True)

    return training_dataloader, testing_dataloader

def split_dataset(dataset):
    """
    Split a torch Dataset into its component examples and labels.
    """
    data = list(zip(*dataset))
    examples = torch.stack(data[0])
    labels = torch.tensor(data[1])
    return examples, labels

def plot_digit_histogram(dataloaders, title='', labels=[]):
    """
    Takes one or more DataLoaders and creates a histogram of the
    occurence of samples in each DataLoader.
    """
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_ylabel('Digit Examples')
    ax.set_xlabel('Digit')
    
    labels_by_dataloader = []
    for dataset in tqdm([dataloader.dataset for dataloader in dataloaders], desc='Tabulating datasets'):
        _, data_labels = split_dataset(dataset)
        labels_by_dataloader.append(sorted(Counter(data_labels)))
        
    H = ax.hist(labels_by_dataloader, bins=range(11), label=labels, histtype='bar', align='left', rwidth=0.8)

def make_custom_datasets(dataset, n_workers=0, allocations=None):
    
    _, y = split_dataset(dataset)
    classes = set(y.numpy())
    n_classes = len(classes)
    
    # default to a square matrix
    n_workers = n_classes if n_workers <= 0 else n_workers
    
    # baseline probabilities are uniform
    probabilities = np.array([[1.0] * n_workers] * n_classes)
    
    # adjust probabilities with allocation tuples
    # make sure to test for array bounds here - those tuples could be anything
    for allocation in allocations:
        #ALLOCATE
        probabilities[allocation[0]][allocation[1]] = allocation[2]
    
    # normalize the matrix on both axes
    for x in range(1000):
        probabilities = np.transpose(probabilities)
        probabilities = np.array([probabilities[i]/probabilities[i].sum() for i in range(len(probabilities))])
        probabilities = np.transpose(probabilities)
        probabilities = np.array([probabilities[i]/probabilities[i].sum() for i in range(len(probabilities))])
    
    # allocate examples to each worker, according to his or her probability 
    # by placing that worker's 'name' in a list as long as the number of examples
    idx_dset = np.array([np.random.choice(n_workers, p=probabilities[yi]) for yi in y])
    
    # for each worker, get a list of the indices of every example allocated to that worker
    dset_idx = [np.where(idx_dset == di)[0] for di in range(n_workers)]
    
    # pull each item from the source dataset and add it to a subset for each worker
    custom_datasets = [Subset(dataset, di) for di in dset_idx]
    
    return custom_datasets

def make_federated_dataloaders(dataset, p=None, batch_size=default_training_batch_size, shuffle=True):
    
    return [torch.utils.data.DataLoader(dataset, batch_size, shuffle) for dataset in make_federated_datasets(dataset, p=p)]
    
def make_federated_datasets(dataset, p=None):
    """
    Allocate the examples in a torch Dataset to exactly one of N torch Subset Datasets, where N is 
    the number of classes in the input Dataset. Parameter p is the overweighting bias of samples of a
    class in the Subset that corresponds to the class.
    """
    _, y = split_dataset(dataset)
    classes = set(y.numpy())
    n_classes = len(classes)
    idx_dset = index_to_dataset(y, p=p)
    dset_idx = [np.where(idx_dset == di)[0] for di in range(n_classes)]
    return [torch.utils.data.Subset(dataset, di) for di in dset_idx]

def index_to_dataset(y, p=None):
    """
    Provides a two-dimensional array of probabilities that a given sample will be allocated to 
    a specific worker. This implementation assumes that the number of workers is equal to the 
    number of classes in the dataset.
    """
    classes = set(y.numpy())
    n_classes = len(classes)
    p = p or 1/n_classes
    pnot = (1-p)/(n_classes-1)
    ps = np.full((n_classes, n_classes), pnot)
    np.fill_diagonal(ps, p)
    return np.array([np.random.choice(10, p=ps[yi]) for yi in y])

def print_dataset_counters(datasets):
    """
    Counts and print the number of examples in each class in a torch Dataset or list of torch Datasets
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        _, labels = split_dataset(dataset)
        print(Counter(labels.numpy()))

def print_training_update(prefix, history, model_id):

    history_length = len(history["test_loss"])

    print('{}\tloss: {:.4f} ({:+.4f})\tacc: {:6.2%} ({:+7.2%})\tmodel: {}'.format(
        prefix,
        history["test_loss"][-1],
        history["test_loss"][-1] - history["test_loss"][max(-2, -history_length)],
        history["test_accuracy"][-1] / 100,
        (history["test_accuracy"][-1] - history["test_accuracy"][max(-2, -history_length)]) / 100,
        str(model_id)[-5:])
    )
