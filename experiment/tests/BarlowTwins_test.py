import torch
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from src.utils.utils import INIconfig
from src.CV.BarlowTwins import BarlowTwins
from src.utils.DataLoader.ImageNet import get_imagenet_training_dataloader
from src.utils.ModelTrainer import SerialBarlowTwinsModelTrainer
from src.utils.OptimizerUtils import LARS


def test_SerialBarlowTwinsModelTrainer():
    config = INIconfig('config.cfg')
    
    train_dataloader = get_imagenet_training_dataloader(config)
    model = BarlowTwins(config)
    bias_list = []
    param_list = []
    for params in model.parameters():
        if params.dim == 1:
            bias_list.append(params)
        else:
            param_list.append(params)
    parameters = [{'params', param_list}, {'params': bias_list}]
    optimizer = LARS(parameters, float(config.BARLOWTWINS['learning_rate']))
    summaryWriter = SummaryWriter()
    SerialBarlowTwinsModelTrainer(config, model, train_dataloader, optimizer, summary_writer=summaryWriter)