"""
Simulate Federated Learning on a single machine in PyTorch
To use, create a FederatedManager and call its `round` method several times.
"""

import torch
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from mnist_utils import split_dataset
from mnist_utils import make_federated_dataloaders
from mnist_utils import print_training_update


class FederatedManager:

    def __init__(self, 
                 name,
                 dataloaders, 
                 test,
                 make_model,
                 loss_fn=torch.nn.CrossEntropyLoss(), 
                 n_epochs=1, 
                 lr=0.02, 
                 momentum=0.5,
                 verbose=False, 
                 device='cpu',
                 *args, **kwargs):
        
        self.dataloaders = dataloaders
        self.n_workers = len(dataloaders)
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        self.name = name
        self.history = {"test_loss": [], "test_accuracy": []}
        self.make_model = make_model
        self.model = self.make_model()
        self.model.train(False)
        self.loss_fn = loss_fn
        self.Xtest, self.ytest = split_dataset(test)
        self.workers = []
        self.device = device
        for i, dl in enumerate(dataloaders):
            self.workers.append(FederatedWorker(i, 
                                                self, 
                                                dl, 
                                                loss_fn,
                                                n_epochs=n_epochs, 
                                                lr=lr, 
                                                momentum=momentum,
                                                verbose=verbose,
                                                device=device,
                                                *args, **kwargs))
        self.worker_loss_histories = [[] for _ in self.workers]

    def round(self):
        """
        Do a round of federated learning:
         - instruct each worker to train and return its model
         - replace the server model the weighted average of the worker models
         - replace the worker models with the server model
        Workers with `participant=False` train but are not included in the
        weighted average and do not receive a copy of the server model.
        """
        updates = [w.train() for w in self.workers]
        self.fedavg(
            [u for u, w in zip(updates, self.workers) if w.participant]
        )
        self.push_model(w for w in self.workers if w.participant)
        self.record_loss()

    def fedavg(self, updates):
        """
        Replace the manager model with the weighted average of the node models.
        """
        N = sum(u["n_samples"] for u in updates)
        for key, value in self.model.state_dict().items():
            weight_sum = (
                u["state_dict"][key] * u["n_samples"] for u in updates
            )
            value[:] = sum(weight_sum) / N

    def push_model(self, workers):
        """
        Push manager model to a list of workers.
        """
        for worker in workers:
            worker.model = self.copy_model()

    def copy_model(self):
        """
        Return a copy of the current manager model.
        """
        model_copy = self.make_model()
        model_copy.load_state_dict(self.model.state_dict())
        return model_copy

    def evaluate_model(self, model=None):
        """
        Compute the loss and accuracy of model on test set.
        """
        model = model or self.model

        #print('\t\teval:', str(id(model))[-5:])
        #print('mgrs:', str(id(self.model))[-5:])
        model = model.to(self.device)

        self.Xtest = self.Xtest.to(self.device)
        self.ytest = self.ytest.to(self.device)

        was_training = model.training
        model.train(False)
        with torch.no_grad():
            output = model(self.Xtest)

            output = output.to(self.device)

            loss = self.loss_fn(output, self.ytest).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(self.ytest.view_as(pred)).sum().item()
        model.train(was_training)

        #print('\t\t', loss, correct)

        return loss, 100. * correct / len(self.ytest)

    def record_loss(self):
        """
        Record loss of manager model and all worker models on test set.
        """
        loss_accuracy = self.evaluate_model()
        self.history["test_loss"].append(loss_accuracy[0])
        self.history["test_accuracy"].append(loss_accuracy[1])


    def learn(self, n_rounds, target_accuracy=None):
    
        target_met = False;
        
        if (target_accuracy):
            print('{} manager training with {} worker(s) for up to {} rounds or {:.2%} accuracy.'.format(
                    self.name, self.n_workers, n_rounds, target_accuracy / 100,))
        else:
            print('{} manager training with {} worker(s) for {} rounds.'.format(
                    self.name, self.n_workers, n_rounds,))

        for i in trange(n_rounds, desc='Rounds'):
            if(self.verbose):
                print('\nRound {:<3} workers:'.format(i))
            self.round()
            if(self.verbose):
                print_training_update('\nRound {:<3} combined:'.format(i), self.history, id(self.model))

            if(target_accuracy and (self.history['test_accuracy'][-1] >= target_accuracy)):
                target_met = True
                break;

        if(target_met):
            print('{} manager stopped: met accuracy target of {:.2%} after {} rounds. (Test accuracy {:.2%} and loss {:.4f}.)'.format(
                    self.name, target_accuracy / 100, len(self.history['test_accuracy']), self.history['test_accuracy'][-1] / 100, self.history['test_loss'][-1],))
        else:
            print('{} manager trained {} rounds. (Test accuracy {:.2%} and loss {:.4f}.)'.format(
                    self.name, len(self.history['test_accuracy']), self.history['test_accuracy'][-1] / 100, self.history['test_loss'][-1],))



class FederatedWorker:

    def __init__(self, 
                 name, 
                 manager, 
                 dataloader, 
                 loss_fn, 
                 n_epochs=1, 
                 lr=0.02,
                 momentum=0.5,
                 participant=True, 
                 verbose=False,
                 device='cpu',
                 ):

        self.name = name
        self.manager = manager
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.participant = participant
        self.model = manager.copy_model()
        self.n_samples = len(self.dataloader.dataset)
        self.history = {"train_loss": [], "test_loss": [], "test_accuracy": []}
        self.lr = lr
        self.momentum = momentum
        self.verbose = verbose
        self.device = device

    def train(self):
        """
        Train for n_epochs, then return the state dictionary of the model and
        the amount of training data used.
        """
        self.model = self.model.to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=1.) # this is used in the example code, but it's slower

        self.model.train(True)
        for epoch in range(self.n_epochs):
            for i, (x, y) in enumerate(self.dataloader):

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                ypred = self.model(x)
                train_loss = self.loss_fn(ypred, y)
                train_loss.backward()
                optimizer.step()
                self.history["train_loss"].append(train_loss.item())

        loss_accuracy = self.manager.evaluate_model(self.model)
        self.history["test_loss"].append(loss_accuracy[0])
        self.history["test_accuracy"].append(loss_accuracy[1])
        
        if(self.verbose):
            print_training_update('\tworker {:2}:'.format(self.name), self.history, id(self.model))

        return {
            "state_dict": self.model.state_dict(),
            "n_samples": self.n_samples
        }


def plot_managers(mgrs, plot_workers=False):
    
    if not isinstance(mgrs, list):
        mgrs = [mgrs]
    
    fig, ax = plt.subplots()
    for m in mgrs:
        ax.plot(m.history['test_loss'], label=m.name)
        if(plot_workers):
            for w in m.workers:
                ax.plot(w.history['test_loss'], label=(m.name, 'Worker ' + str(w.name)))
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.legend();
    
    fig, ax = plt.subplots()
    for m in mgrs:
        ax.plot(m.history['test_accuracy'], label=m.name)
        if(plot_workers):
            for w in m.workers:
                ax.plot(w.history['test_accuracy'], label=(m.name, 'worker ' + str(w.name)))
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.legend();


def evaluate_new_manager(name, training_dataset, testing_dataset, p=0.0, n_rounds=50, target_accuracy=None, model=None, verbose=False):
    
    dataloaders = make_federated_dataloaders(training_dataset, p=p)
    manager = FederatedManager(name, dataloaders, testing_dataset, model, verbose=verbose)
    manager.learn(n_rounds, target_accuracy)
    plot_managers(manager)
    return manager

