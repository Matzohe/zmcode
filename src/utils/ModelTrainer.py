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
from .OptimizerUtils import LARS_adjust_learning_rate, LARS_adjust_learning_rate_normal

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

    before_batch_idx = 0
    before_epoch_idx = 0
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
        for batch_num, ((_x, _y), _) in tqdm(enumerate(train_dataloader, start=before_batch_idx), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            # TODO: add Image preprocess
            serial_num += 1
            image1 = _x.to(config.TRAINING['device'])
            image2 = _y.to(config.TRAINING['device'])
            loss = model(image1, image2)
            loss.backward()
            LARS_adjust_learning_rate(config, optimizer, train_dataloader, batch_num + epoch * len(train_dataloader))
            all_batch_loss += loss.detach().cpu()
            if serial_num < serial_size:
                if batch_num == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    if summary_writer is not None:
                        summary_writer.add_scalar("Loss", all_batch_loss / serial_num, epoch * len(train_dataloader) + batch_num )
                    serial_num = 0
                    all_batch_loss = 0
                continue

            optimizer.step()
            optimizer.zero_grad()

            # write training loss to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar("Loss", all_batch_loss / serial_num, epoch * len(train_dataloader) + batch_num )
            all_batch_loss = 0
            serial_num = 0
            end_time = time.perf_counter()

            # save training check point
            if end_time - start_time > int(config.TRAINING["checkpoint_save_time"]):
                save_checkpoint(config, model, optimizer, batch_num, epoch)
                start_time = time.perf_counter()
        before_batch_idx = 0
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
    checkpoint_path: str = None,
    true_batch_size: int = None
):
    # This function has a limitation, the model's output should be the same as the label
    # TODO: Change the data structure to fp16 while training.
    epochs = int(config.MODEL['epoch'])
    before_batch_idx = 0
    before_epoch_idx = 0
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
    for epoch in range(before_epoch_idx,epochs):
        if from_checkpoint and before_epoch_idx > epoch:
            continue
        start_time = time.perf_counter()
        dataloader_batch_size = train_dataloader.batch_size
        if true_batch_size is None:
            true_batch_size = int(config.MODEL['batch_size'])
        else:
            assert (true_batch_size >= dataloader_batch_size or true_batch_size % dataloader_batch_size == 0)
            serial_size = true_batch_size // dataloader_batch_size  # use for serial training


        all_batch_loss = 0
        serial_num = 0
        for batch_num, (_x, _y) in tqdm(enumerate(train_dataloader, start=before_batch_idx), desc=f"Epoch {epoch}", total=len(train_dataloader)):
            serial_num += 1
            _x = _x.to(config.TRAINING['device'])
            _y = _y.to(config.TRAINING['device'])
            
            output_y = model(_x)
            loss = loss_fn(output_y, _y)
            optimizer.zero_grad()
            loss.backward()

            LARS_adjust_learning_rate(config, optimizer, train_dataloader, batch_num + epoch * len(train_dataloader))
            all_batch_loss += loss.detach().cpu()

            if serial_num < serial_size:
                if batch_num == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    if summary_writer is not None:
                        summary_writer.add_scalar("Loss", all_batch_loss / serial_num, epoch * len(train_dataloader) + batch_num )
                    serial_num = 0
                    all_batch_loss = 0
                continue

            optimizer.step()
            optimizer.zero_grad()


            # write training loss to tensorboard
            if summary_writer is not None:
                summary_writer.add_scalar("Loss", all_batch_loss / serial_num, epoch * len(train_dataloader) + batch_num)
            all_batch_loss = 0
            serial_num = 0

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

        before_batch_idx = 0

    # save trained model
    model_state_dict = model.state_dict()
    save_root = config.TRAINING["model_save_root"].format(type(model).__name__)
    if not os.path.exists("/".join(save_root.split('/')[:-1])):
        os.makedirs("/".join(save_root.split('/')[:-1]))
    torch.save(model_state_dict, save_root)
    print("Model saved at {}".format(save_root))