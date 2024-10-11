# Target: A model trainer, which need a model, a training dataset, a validation dataset, a optimizer, a loss function, and a config

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List, Dict
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import time
from .ImagePreporcessUitls import BarlowTwinsTransform
from .CheckPointUtils import save_checkpoint

def SerialBarlowTwinsModelTrainer(
    config,
    model: nn.Module,
    train_dataloader,
    optimizer: torch.optim.Optimizer,
    val_dataloader = None,
    summary_writer: Union[SummaryWriter] = None,
    from_checkpoint: bool = False,
    checkpoint_path: str = None):

    epochs = int(config.MODEL['epoch'])
    dataloader_batch_size = train_dataloader.batch_size
    training_batch_size = int(config.BARLOWTWINS['batch_size'])
    serial_size = training_batch_size // dataloader_batch_size  # use for serial training

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
    
    # TODO: change to multi-gpu training
    for epoch in range(before_epoch_idx, epochs):
        start_time = time.perf_counter()

        serial_num = 0
        all_batch_loss = 0
        for batch_num, (_x, _) in tqdm(enumerate(train_dataloader, start=before_batch_idx), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            # TODO: add Image preprocess
            serial_num += 1
            _x = _x.to(config.TRAINING['device'])
            image1 = BarlowTwinsTransform(_x)
            image2 = BarlowTwinsTransform(_x)
            loss = model(image1, image2)
            loss.backward()
            if serial_num < serial_size:
                all_batch_loss += loss.detach().cpu()
                continue
            else:
                serial_num = 0
                all_batch_loss += loss.detach().cpu()
            optimizer.step()
            optimizer.zero_grad()

            # write training loss to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar("Loss", all_batch_loss, epoch * len(train_dataloader) + batch_num)
            all_batch_loss = 0
            end_time = time.perf_counter()

            # save training check point
            if end_time - start_time > int(config.TRAINING["checkpoint_save_time"]):
                save_checkpoint(config, model, optimizer, batch_num, epoch)
                start_time = time.perf_counter()
    # save trained model
    model_state_dict = model.state_dict()
    save_root = config.TRAINING["model_save_root"].format(type(model).__name__)
    if not os.path.exists("/".join(save_root.split('/')[:-1])):
        os.makedirs("/".join(save_root.split('/')[:-1]))
    torch.save(model_state_dict, save_root)
    print("Model saved at {}".format(save_root))



# TODO: Add UnsupervisedModelTrainer
def BasicSupervisedModelTrainer(
    config,
    model: nn.Module,
    train_dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    val_dataloader = None,
    summary_writer: Union[SummaryWriter] = None,
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
            
            output_y = model(_x)
            loss = loss_fn(output_y, _y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # write training loss to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar("Loss", loss, epoch * len(train_dataloader) + batch_num)
            end_time = time.perf_counter()

            # save training check point
            if end_time - start_time > int(config.TRAINING["checkpoint_save_time"]):
                save_checkpoint(config, model, optimizer, batch_num, epoch)
                start_time = time.perf_counter()
        if val_dataloader is not None:
            all_acc = 0
            all_num = 0
            with torch.no_grad():
                for val_batch_num, (_x, _y) in tqdm(enumerate(val_dataloader), desc=f"Epoch {epoch}", total=len(val_dataloader)):
                    _x = _x.to(config.TRAINING['device'])
                    _y = _y.to(config.TRAINING['device'])
                    output_y = model(_x)
                    all_acc += (output_y.argmax(1) == _y).sum()
                    all_num += len(_y)
            if summary_writer is not None:
                summary_writer.add_scalar("Accuracy", all_acc / all_num, epoch)
            else:
                print("Epoch:{}, Accuracy: {}".format(epoch, all_acc / all_num))

    # save trained model
    model_state_dict = model.state_dict()
    save_root = config.TRAINING["model_save_root"].format(type(model).__name__)
    if not os.path.exists("/".join(save_root.split('/')[:-1])):
        os.makedirs("/".join(save_root.split('/')[:-1]))
    torch.save(model_state_dict, save_root)
    print("Model saved at {}".format(save_root))