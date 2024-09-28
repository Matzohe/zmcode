# -*- coding='utf-8' -*-
import numpy as np
import random
import jieba
import os
from torch.utils.data import DataLoader, Dataset
from gensim.models.keyedvectors import KeyedVectors
import pickle

stopwords = []


def mystrip(ls):
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    _i = 0
    for _ in range(len(_words)):
        if _words[_i] in stopwords or _words[_i].strip() == "":
            # print(_words[_i])
            _words.pop(_i)
        else:
            _i += 1
    return _words

def load_data(config):
    jieba.setLogLevel(jieba.logging.INFO)

    pos_sentence, pos_label, neg_sentence, neg_label = [], [], [], []

    pos_fname = os.path.join(config.DATASET["hotel_emotion"], "positive")
    neg_fname = os.path.join(config.DATASET["hotel_emotion"], "negative")

    for f_name in os.listdir(pos_fname):
        with open(pos_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)
            pos_sentence.append(remove_stopwords(words))

            pos_label.append(1)  # label为1表示积极，label为0表示消极
            
    for f_name in os.listdir(neg_fname):
        with open(neg_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)
            neg_sentence.append(remove_stopwords(words))
            neg_label.append(0)

    return pos_sentence, pos_label, neg_sentence, neg_label

def string2vec(config, word_vectors, sentence):
    MAX_SENT_LEN = int(config.HOTEL['max_sent_len'])
    WORD_DIM = int(config.HOTEL['word_dim'])

    for i in range(len(sentence)):
        sentence[i] = sentence[i][:MAX_SENT_LEN]
        line = sentence[i]
        for j in range(len(line)):
            if line[j] in word_vectors:
                line[j] = word_vectors.get_vector(line[j])
            else:
                line[j] = np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32")
        if len(line) < MAX_SENT_LEN:
            for k in range(MAX_SENT_LEN-len(line)):
                sentence[i].append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))
    return sentence


class Corpus(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def get_data(config):
    TRAIN_SPLIT = float(config.HOTEL['train_split'])

    global stopwords
    with open(config.DATASET['stopwords'], encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)

    pos_sentence, pos_label, neg_sentence, neg_label = load_data(config)
    sentence = pos_sentence + neg_sentence
    label = pos_label + neg_label

    sentence = sentence[:]
    label = label[:]

    shuffle = list(zip(sentence, label))
    random.shuffle(shuffle)
    sentence[:], label[:] = zip(*shuffle)

    assert len(sentence) == len(label)
    length = int(TRAIN_SPLIT*len(sentence))
    train_sentence = sentence[:length]
    train_label = label[:length]
    test_sentence = sentence[length:]
    test_label = label[length:]

    # 加载词向量
    print("loading word2vec...")
    word_vectors = KeyedVectors.load_word2vec_format(config.HOTEL['word2vec'])
    print("loading end")

    # 将string单词转为词向量
    train_sentence = string2vec(config, word_vectors, train_sentence)
    test_sentence = string2vec(config, word_vectors, test_sentence)

    # 拼接一句话中的所有词向量（可根据要求改为对所有词向量求和）
    train_sentence = [np.concatenate(wordvecs) for wordvecs in train_sentence]
    test_sentence = [np.concatenate(wordvecs) for wordvecs in test_sentence]

    # 生成数据集
    train_set = Corpus(train_sentence, train_label)
    test_set = Corpus(test_sentence, test_label)

    return train_set, test_set

def generate_and_save_data(config):

    train_set, test_set = get_data(config)
    outpath = os.path.join(config.DATASET['hotel_emotion'], 'train_set.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(train_set, f)
    outpath = os.path.join(config.DATASET['hotel_emotion'], 'test_set.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(test_set, f)
