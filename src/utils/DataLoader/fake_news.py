import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class FakeNewsDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.rate = float(config.BERTCLASSIFICATION['rate'])
        self.data = pd.read_csv(config.DATASET['fake_news'])
        if train:
            self.data = self.data[: int(len(self.data) * self.rate)]
        else:
            self.data = self.data[int(len(self.data) * self.rate): ]
        self.title_list = []
        self.official_list = []
        self.comment_list = []
        self.label = []
        for i in range(len(self.data)):
            self.title_list.append(self.data.iloc[i]['Title'])
            self.official_list.append(self.data.iloc[i]['Ofiicial Account Name'])
            self.comment_list.append(self.data.iloc[i]['Report Content'])
            self.label.append(int(self.data.iloc[i]['label']))

    def __getitem__(self, index):
        return (self.official_list[index], self.title_list[index], self.comment_list[index]), self.label[index]

    def __len__(self):
        return len(self.data)
    

def fakeNewsTrainingDataLoader(config):
    dataset = FakeNewsDataset(config, train=True)
    return DataLoader(dataset, batch_size=int(config.BERTCLASSIFICATION['batch_size']), shuffle=True)


def fakeNewsValidationDataLoader(config):
    dataset = FakeNewsDataset(config, train=False)
    return DataLoader(dataset, batch_size=int(config.BERTCLASSIFICATION['batch_size']), shuffle=True)
