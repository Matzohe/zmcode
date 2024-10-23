from src.utils.utils import INIconfig
from src.utils.DataLoader.fake_news import fakeNewsTrainingDataLoader, fakeNewsValidationDataLoader
from src.NLP.BertClassification import FakeNewsBertClassification
from src.utils.ModelTrainer import BasicSupervisedModelTrainer
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def fakeNewsTest():
    config = INIconfig()
    device = config.BERTCLASSIFICATION['device']
    model = FakeNewsBertClassification(config)
    training_dataloader = fakeNewsTrainingDataLoader(config)
    val_dataloader = fakeNewsValidationDataLoader(config)
    optimizer = optim.Adam(model.parameters(), lr=float(config.BERTCLASSIFICATION['lr']), betas=(0.9, 0.95))
    epochs = int(config.BERTCLASSIFICATION['epochs'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_num, (_x, _y) in tqdm(enumerate(training_dataloader), desc=f"Epoch {epoch}", total=len(training_dataloader)):  
            
            output_y = model(_x)
            loss = loss_fn(output_y, _y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            print("Epoch:{}, Accuracy: {}".format(epoch, all_acc / all_num))
