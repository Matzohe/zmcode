from src.CV.ResNet50 import ResNet
from src.utils.OptimizerUtils import configure_optimizers
from src.utils.ModelTrainer import BasicSupervisedModelTrainer
from src.utils.utils import set_seed, INIconfig
from src.utils.DataLoader.ImageNet import get_imagenet_training_dataloader, get_imagenet_validation_dataloader
from src.utils.ModelInitializeUtils import CNN_initialize
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def Imagenet_train_resnet50_test():
    config = INIconfig('config.cfg')
    model = ResNet(config).to(device=config.TRAINING['device'])
    train_dataloader = get_imagenet_training_dataloader(config)
    val_dataloader = get_imagenet_validation_dataloader(config)
    optimizer = configure_optimizers(model, config)
    loss_fn = nn.CrossEntropyLoss()
    summary_writer = SummaryWriter()
    BasicSupervisedModelTrainer(config=config, model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_fn=loss_fn, summary_writer=summary_writer, val_dataloader=val_dataloader)