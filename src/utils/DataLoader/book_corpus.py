import pyarrow.parquet as pq
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.NLP.tokenizer.one_hot import OneHot
import src.MultiModal.clip as clip


class BookCorpusDataset(Dataset):
    def __init__(self, config, index):
        self.config = config
        self.book_corpus_root = config.DATASET["book_corpus_root"].format(index)
        self.data = pq.ParquetDataset(self.book_corpus_root).read().to_pydict()["text"][:100000]
        self.tokenizer = OneHot()


    def __getitem__(self, index):
        tokenized = self.tokenizer.encode(self.data[index])
        return tokenized[: -1], tokenized[1:]

    def __len__(self):
        return len(self.data)
    

def get_book_corpus_dataloader(config):
    dataset = BookCorpusDataset(config, index=0)
    dataloader = DataLoader(dataset, batch_size=int(config.TRAINING['batch_size']), shuffle=False)
    return dataloader
