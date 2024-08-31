# Target: A model trainer, which need a model, a training dataset, a validation dataset, a optimizer, a loss function, and a config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# TODO: Add UnsupervisedModelTrainer
def SupervisedModelTrainer(
    config,
    model: nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    summary_writer: torch.utils.tensorboard.SummaryWriter
):
    # This function has a limitation, the model's output should be the same as the label
    epochs = int(config.MODEL['epoch'])

    for epoch in range(epochs):
        pass
    