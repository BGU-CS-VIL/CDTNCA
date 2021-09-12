import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer():
    def __init__(self, args, model, train_loader, val_loader, optimizer, device):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.sched = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=30, verbose=True)

    def train(self):
        """
        Args:
        Returns:
            trained model (nn.module)
        """

        # Train
        min_loss = np.inf
        train_loop = tqdm(range(1, self.args.epochs + 1), desc="Train")
        for epoch in train_loop:
            train_loss = self.train_epoch()
            val_loss = self.test(self.val_loader)
            self.sched.step(val_loss)
            # save checkpoint
            if val_loss < min_loss:
                min_loss = val_loss
                self._save_checkpoint(val_loss, self.args.exp_name)
            train_loop.set_postfix(train_loss=train_loss.item(), val_loss=val_loss.item())

        # Load best model based on validation loss
        checkpoint = torch.load(f'{self.args.exp_name}_checkpoint.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self.model

    def train_epoch(self):
        self.model.train()
        running_loss = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(self.device)
            self.optimizer.zero_grad()
            output, thetas = self.model(data)
            loss = self.model.loss(output, target)
            running_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()
        return np.mean(running_loss)

    def test(self, loader):
        self.model.eval()
        with torch.no_grad():
            loss = []
            for data, target in loader:
                data, target = data.type(torch.FloatTensor).to(self.device), target.type(torch.FloatTensor).to(
                    self.device)
                output, theta = self.model(data)
                loss.append(self.model.loss(output, target).item())
        return np.mean(loss)

    def _save_checkpoint(self, test_loss, exp_name=''):
        """
        Args:
            test_loss: float, test loss at time of saving the checkpoint
            exp_name: file name (str)
        """
        checkpoint = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': test_loss
                      }

        torch.save(checkpoint, f'{exp_name}_checkpoint.pth')
