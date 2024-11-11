from src.utils.utils import INIconfig
from collections import OrderedDict
from tqdm import tqdm
import torch


def word_freq_count(sentences, vocab):
    for word in sentences.split(" "):
        vocab[word] = vocab.get(word, 0) + 1
    
def count_book_corpus():
    # config = INIconfig("config/rnn_config.cfg")
    # dataset = BookCorpusDataset(config, index=0)
    # vocab = OrderedDict()
    # for i in tqdm(range(len(dataset))):
    #     word_freq_count(dataset[i], vocab)

    # info_list = list(vocab.items())
    # info_list.sort(key=lambda x: x[1], reverse=True)
    # torch.save(info_list, "experiment/output/RNN/info_list.pt")

    info_list = torch.load("experiment/output/RNN/info_list.pt")
    final_dict = OrderedDict()
    for each in info_list:
        if each[1] > 1000:
            final_dict[each[0]] = each[1]
    
    torch.save(final_dict, "experiment/output/RNN/final_dict.pt")


class OneHot:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = torch.load("experiment/output/RNN/final_dict.pt")
        
        self.vocab = vocab
        self.word2idx = OrderedDict({w: i + 4 for i, w in enumerate(self.vocab.keys())})
        self.word2idx["<unk>"] = 0
        self.word2idx["<pad>"] = 1
        self.word2idx["<s>"] = 2
        self.word2idx["</s>"] = 3

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text, padding=True, max_len=128):
        text = text.split(" ")
        encode_ids = [2]
        for word in text:
            encode_ids.append(self.word2idx.get(word, 0))
        encode_ids.append(3)
        if padding:
            while len(encode_ids) < max_len:
                encode_ids.append(0)
            if len(encode_ids) > max_len:
                encode_ids = encode_ids[:max_len]

        return torch.tensor(encode_ids)
    
    def decode(self, ids):
        text = []
        for id in ids:
            text.append(self.idx2word[id])
        return " ".join(text)