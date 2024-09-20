# Target: A model trainer, which need a model, a training dataset, a validation dataset, a optimizer, a loss function, and a config

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Union, List, Dict

import os
from tqdm import tqdm
import time

from .CheckPointUtils import save_checkpoint

# TODO: Add UnsupervisedModelTrainer
def BasicSupervisedModelTrainer(
    config,
    model: nn.Module,
    train_dataloader: torch.utils.data.Dataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    summary_writer: Union[torch.utils.tensorboard.SummaryWriter] = None,
    from_checkpoint: bool = False,
    checkpoint_path: str = None
):
    # This function has a limitation, the model's output should be the same as the label
    # TODO: Change the data structure to fp16 while training.
    epochs = int(config.MODEL['epoch'])
    
    if from_checkpoint:
        try:
            checkpoint = torch.load(checkpoint_path)
        except:
            raise RuntimeError("Something wrong with checkpoint loading, please check your checkpoint path")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        before_batch_idx = checkpoint['batch_idx']
        before_epoch_idx = checkpoint['epoch_idx']
        print("Checkpoint loaded at {}".format(checkpoint_path))
    
    # TODO: Shuffle the training datasets at the start of the epoch, need to change the training logics
    for epoch in range(epochs):
        if from_checkpoint and before_epoch_idx > epoch:
            continue
        start_time = time.perf_counter()
        for batch_num, (_x, _y) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            _x = _x.to(config.TRAINING['device'])
            _y = _y.to(config.TRAINING['device'])
            if from_checkpoint and before_batch_idx > batch_num:
                continue
            if summary_writer is not None:
                if not os.path.exists(config.TRAINING["log_dir"]):
                    try:
                        os.mkdir(config.TRAINING["log_dir"])
                    except:
                        raise ValueError("Can't create log dir, you need to create the root path before create the log file")
            output_y = model(_x)
            loss = loss_fn(output_y, _y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write training loss to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar("loss", loss, epoch)
            end_time = time.perf_counter()

            # save training check point
            if end_time - start_time > int(config.TRAINING["checkpoint_save_time"]):
                save_checkpoint(config, model, optimizer, batch_num, epoch)
                start_time = time.perf_counter()