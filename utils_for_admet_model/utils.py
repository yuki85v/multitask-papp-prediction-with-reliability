import numpy as np
import torch
from torch import nn


def fix_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopping:
    '''
    Earlystopping for PyTorch implementations

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): If True, prints a message for each validation loss improvement.
        save_model (bool): If True, saves the model when the validation loss improves.
        save_path (list): List of paths to save the models.

    Attributions:
        counter: Counts the number of epochs since the last improvement.
        best_score: Best score (validation loss) achieved so far.
        early_stop: Boolean flag indicating whether to stop training early.
        val_loss_min: Minimum validation loss achieved so far.
    '''

    def __init__(self, patience=10, verbose=False, save_model=False, save_path=None):
        self.patience = patience
        self.verbose = verbose
        self.save_model = save_model
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'{self.counter}/{self.patience}: val_loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if self.save_model:
            for i, func in enumerate(model):
                torch.save(func.state_dict(), self.save_path[i])
        self.val_loss_min = val_loss


class MultiTaskLoss(nn.Module):
    def __init__(self, weighted=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.weighted = weighted

    def forward(self, pred, label):
        # Create a mask for non-NaN values in the label
        mask = ~torch.isnan(label)
        label = torch.nan_to_num(label * mask)
        pred = pred * mask

        if self.weighted:
            # Compute the non-zero ratio for each task
            nonzero_ratio = torch.count_nonzero(label, dim=0) / label.shape[0]
            total_loss = 0
            for task_idx in range(label.shape[1]):

                num_data = mask[:, task_idx].sum()
                if nonzero_ratio[task_idx] != 0:
                    loss = self.mse(pred[:, task_idx], label[:, task_idx]) / nonzero_ratio[task_idx]
                    loss /= num_data
                else:
                    loss = 0
                total_loss += loss
        else:
            # Compute the non-zero ratio for the entire batch
            nonzero_ratio = torch.count_nonzero(label) / label.numel()
            total_loss = self.mse(pred, label) / nonzero_ratio

        return total_loss