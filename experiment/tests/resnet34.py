import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.utils.utils import set_seed, INIconfig
from src.utils.DataLoader.cifar import data_loader
from src.CV.ResNet34 import ResNet34
from src.utils.ModelTrainer import BasicSupervisedModelTrainer
from src.utils.OptimizerUtils import configure_optimizers


def Resnet34Test():
    set_seed(42)
    config = INIconfig("config.cfg")
    model = ResNet34(config)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    training_dataset, _ = data_loader(config)
    summary_writer = SummaryWriter()
    BasicSupervisedModelTrainer(config, model, training_dataset, optimizer, loss_fn, summary_writer=summary_writer)
    