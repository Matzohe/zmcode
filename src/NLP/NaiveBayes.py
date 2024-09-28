import jieba
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

class NaiveBayes:
    """
    朴素贝叶斯分类器
    """

    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(config.DATASET["hotel_emotion"], "train.txt")
        self.test_path = os.path.join(config.DATASET["hotel_emotion"], "test.txt")
        
        self.sum_words_neg = 0
        self.sum_words_pos = 0
        
        self.neg_sents_train = []
        self.pos_sents_train = []
        self.neg_sents_test = []
        self.pos_sents_test = []
        
        self.stopwords = []
        self._neg_dict = OrderedDict([])
        self._pos_dict = OrderedDict([])

        self.initialize()


    def mystrip(self, ls):
        # remove the \n in the end of a sentence
        for i in range(len(ls)):
            ls[i] = ls[i].strip("\n")
        return ls

    def remove_stopwords(self, _words):
        # remove the stopwords
        _i = 0
        for _ in range(len(_words)):
            if _words[_i] in self.stopwords:
                _words.pop(_i)
            else:
                _i += 1
        return _words

    def initialize(self):
        with open(self.config.DATASET['stopwords'], encoding='utf-8') as f:
            self.stopwords = f.readlines()
            self.stopwords = self.mystrip(self.stopwords)
        
        with open(self.train_path, encoding='utf-8') as f:
            print("loading training dataset...")
            for line in f:
                line = line.strip('\n')
                if line[0] == '1':
                    self.pos_sents_train.append(line[1:])
                else:
                    self.neg_sents_train.append(line[1:])

        with open(self.test_path, encoding='utf-8') as f:
            print("loading testing dataset...")
            for line in f:
                line = line.strip('\n')
                if line[0] == '1':
                    self.pos_sents_test.append(line[1:])
                else:
                    self.neg_sents_test.append(line[1:])

        
        for i in tqdm(range(len(self.neg_sents_train)), desc="generating negative dictionary..."):
            neg_words = jieba.lcut(self.neg_sents_train[i])
            neg_words = self.remove_stopwords(neg_words)
            for each in neg_words:
                self._neg_dict[each] = self._neg_dict.get(each, 0) + 1
        
        for i in tqdm(range(len(self.pos_sents_train)), desc="generating positive dictionary..."):
            pos_words = jieba.lcut(self.pos_sents_train[i])
            pos_words = self.remove_stopwords(pos_words)
            for each in pos_words:
                self._pos_dict[each] = self._pos_dict.get(each, 0) + 1
        
        self.sum_words_neg = len(self._neg_dict)
        self.sum_words_pos = len(self._pos_dict)
    
    def predict(self):
        acc_num = 0
        for i in tqdm(range(len(self.neg_sents_test)), desc="predicting negative datasets..."):
            st = jieba.lcut(self.neg_sents_test[i])
            st = self.remove_stopwords(st)
            p_neg = 0
            p_pos = 0
            for word in st:
                if word in self._neg_dict:
                    p_neg += np.log(self._neg_dict[word] / self.sum_words_neg)
                else:
                    p_neg += np.log(1 / self.sum_words_neg)
                if word in self._pos_dict:
                    p_pos += np.log(self._pos_dict[word] / self.sum_words_pos)
                else:
                    p_pos += np.log(1 / self.sum_words_pos)

            if p_neg > p_pos:
                acc_num += 1
        for i in tqdm(range(len(self.pos_sents_test)), desc="predicting positive datasets..."):
            st = jieba.lcut(self.pos_sents_test[i])
            st = self.remove_stopwords(st)
            p_neg = 0
            p_pos = 0
            for word in st:
                if word in self._neg_dict:
                    p_neg += np.log(self._neg_dict[word] / self.sum_words_neg)
                else:
                    p_neg += np.log(1 / self.sum_words_neg)
                if word in self._pos_dict:
                    p_pos += np.log(self._pos_dict[word] / self.sum_words_pos)
                else:
                    p_pos += np.log(1 / self.sum_words_pos)

            if p_pos > p_neg:
                acc_num += 1

        print("accurate rate:", acc_num / (len(self.neg_sents_test) + len(self.pos_sents_test)))
