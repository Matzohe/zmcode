from src.utils.DataLoader.book_corpus import BookCorpusDataset
from src.utils.utils import INIconfig
from collections import OrderedDict
from tqdm import tqdm


def word_freq_count(sentences, vocab):
    for word in sentences.split(" "):
        vocab[word] = vocab.get(word, 0) + 1
    
def count_book_corpus():
    config = INIconfig("config/rnn_config.cfg")
    dataset = BookCorpusDataset(config, index=0)
    vocab = OrderedDict()
    for i in tqdm(range(10000)):
        word_freq_count(dataset[i], vocab)

    info_list = list(vocab.items())
    info_list.sort(key=lambda x: x[1], reverse=True)
    print(len(info_list))
        
