"""
Simulate Federated Learning on a single machine in PyTorch

To use, create a FederatedManager and call it's `round` method several times.
"""
import torch
import torch.nn
import torch.optim


class FederatedManager:

    def __init__(self, dataloaders, model, loss_fn, lr, test_dset,
                 n_epochs, *args, **kwargs):
        self.n_workers = len(dataloaders)
        self.n_epochs = n_epochs
        self.manager_loss_history = []
        #self.make_model = make_model
        self.model = model
        self.model.train(False)
        self.loss_fn = loss_fn
        self.lr = lr

        Xtest, ytest = list(zip(*test_dset))

        self.Xtest = torch.stack(Xtest)
        self.ytest = torch.LongTensor(ytest)
        self.workers = [
            FederatedWorker(self, dl, loss_fn, lr, n_epochs, *args, **kwargs)
            for dl in dataloaders
        ]
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
        self.record_loss()
        self.fedavg([u for u, w in zip(updates, self.workers) if w.participant])
        self.push_model(w for w in self.workers if w.participant)
        self.record_loss()

    def fedavg(self, updates):
        """
        Replace the manager model with the weighted average of the node models.
        """
        N = sum(u["n_samples"] for u in updates)
        for key, value in self.model.state_dict().items():
            weight_sum = (u["state_dict"][key] * u["n_samples"] for u in updates)
            value[:] = sum(weight_sum) / N

    def push_model(self, workers):
        """
        Pust manager model to a list of workers.
        """
        for worker in workers:
            worker.model = self.copy_model()

    def copy_model(self):
        """
        Return a copy of the current manager model.
        """
        model_copy = self.model
        model_copy.load_state_dict(self.model.state_dict())
        return model_copy

    def evaluate_loss(self, model=None):
        """
        Compute the loss of model on test set.
        """
        model = model or self.model
        was_training = model.training
        model.train(False)
        with torch.no_grad():
            loss = self.loss_fn(model(self.Xtest), self.ytest).item()
        model.train(was_training)
        return loss

    def record_loss(self):
        """
        Record loss of manager model and all worker models on test set.
        """
        self.manager_loss_history.append(self.evaluate_loss())
        for i, worker in enumerate(self.workers):
            worker_loss = self.evaluate_loss(model=worker.model)
            self.worker_loss_histories[i].append(worker_loss)


class FederatedWorker:

    def __init__(
        self, manager, dataloader, loss_fn, lr, n_epochs, participant=True
    ):
        self.manager = manager
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.participant = participant
        self.model = manager.copy_model()
        self.n_samples = len(self.dataloader.dataset)
        self.loss_history = {"train": [], "test": []}
        self.lr = lr

    def train(self):
        """
        Train for n_epochs, then return the state dictionary of the model and
        the amount of training data used.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train(True)
        for epoch in range(self.n_epochs):
            print("    Worker:", id(self) % 1000, "Epoch: ", epoch)
            for i, (x, y) in enumerate(self.dataloader):
                optimizer.zero_grad()
                ypred = self.model(x)
                train_loss = self.loss_fn(ypred, y)
                train_loss.backward()
                optimizer.step()
                self.loss_history["train"].append(train_loss.item())
                # below is expensive, so only do it once per round
                # self.loss_history["test"].append(self.manager.evaluate_loss(self.model))
                if i%100==0: 
                    print("        Worker:", id(self) % 1000, "Batch: %03d" % i, "Loss: %.4f" % self.loss_history["train"][-1])

        return {
            "state_dict": self.model.state_dict(),
            "n_samples": self.n_samples
        }