# 写一个BPE分词器
import torch
import os
import re
from typing import List, Union

# The main function of BPE should realize three function, one is encoder, the other is decoder..
# It's vocab could be a root, or a list
class BPE(object):
    def __init__(self, vocab: Union[str, List[str]]):
        
        if isinstance(vocab, str):
            if not os.path.exists(vocab):
                raise FileNotFoundError("vocab file not found")
            self.vocab = self.load_vocab(vocab)
        elif isinstance(vocab, list):
            self.vocab = vocab
        else:
            raise ValueError("vocab should be a str or a list")

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.eow_token = "</w>"  # The end of a word

        assert self.unk_token in self.word2idx, "unk_token should be in vocab"
        assert self.bos_token in self.word2idx, "bos_token should be in vocab"
        assert self.eos_token in self.word2idx, "eos_token should be in vocab"
        assert self.pad_token in self.word2idx, "pad_token should be in vocab"

        self.unk_token_id = self.word2idx[self.unk_token]
        self.bos_token_id = self.word2idx[self.bos_token]
        self.eos_token_id = self.word2idx[self.eos_token]
        self.pad_token_id = self.word2idx[self.pad_token]
    
    def load_vocab(self, root):
        """_summary_

        Args:
            root (_type_): vocab txt file root

        Returns:
            List[str]: list of vocab
        """
        vocab = []
        with open(root, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                vocab.append(line.strip())
        return vocab

    def encode(self, text: str, bos=True, eos=True) -> List[int]:
        text = self.whitespace_split(text)

        encode_ids = []
        for word in text:
            chars = list(word)
            start = 0
            while start < len(word):
                end = len(word)
                cur_substr = None
                while start < end:
                    current_substr = "".join(chars[start:end])
                    if current_substr in self.word2idx:
                        cur_substr = current_substr
                        break
                    end -= 1
                if cur_substr is None:
                    encode_ids.append(self.unk_token_id)
                    start += 1
                else:
                    encode_ids.append(self.word2idx[cur_substr])
                    start = end
            encode_ids.append(self.eow_token_id)

        if bos:
            encode_ids = [self.bos_token_id] + encode_ids
        if eos:
            encode_ids = encode_ids + [self.eos_token_id]
        return encode_ids

    def whitespace_split(self, text: str) -> List[str]:
        
        text = text.strip()
        if not text:
            return []
        text = text.split()
        return text

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.eow_token_id:
                break
            words.append(self.idx2word[i])
        return " ".join(words)

    def training(self, root: str, vocab_size: int):
        # read training data and return a vocab list

        # Get the training data and put it into a list
        text = []
        with open(root, 'r', encoding="utf-8") as f:
            text_lines = f.readlines()
            text.append(text_lines.strip())

        # Count the frequency of each word
        word_freq = {}
        for sentence in text:
            for word in sentence.split():
                # process the word into a tuple     
                word = tuple(word)
            
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        
        # Initialize the vocab list
        vocab_list = []

        # Get the char pair frequency
        while len(vocab_list) < vocab_size:
            # culculate the word frequency
            char_pair_freq = {}
            for word in word_freq:
                if len(word) < 2:
                    if word not in vocab_list:
                        vocab_list.append(word)
                    continue

                for i in range(len(word) - 1):
                    char_pair = (word[i], word[i+1])
                    if char_pair not in char_pair_freq:
                        char_pair_freq[char_pair] = word_freq[word]
                    else:
                        char_pair_freq[char_pair] += word_freq[word]

            # sort the char pair frequency
            sorted_char_pair_freq = sorted(char_pair_freq.items(), key=lambda x: x[1], reverse=True)

            # select the most frequent char pair
            char_pair = sorted_char_pair_freq.items()[0][0]
            vocab_list.append("".join(char_pair))

            new_word_freq = {}
            # update the word frequency
            for word in word_freq:
                new_word = []
                if len(word) < 2:
                    continue
                
                for i in range(len(word)):

                    if i == len(word) - 1:
                        new_word.append(word[i])
                        continue
                    if word[i] == char_pair[0] and word[i + 1] == char_pair[1]:
                        new_word.append("".join(char_pair))
                        i = i + 1
                    else:
                        new_word.append(word[i])

                new_word = tuple(new_word)
                if new_word not in new_word_freq:
                    new_word_freq[new_word] = word_freq[word]
                else:
                    new_word_freq[new_word] += word_freq[word]
            word_freq = new_word_freq

        return vocab_list
        

if __name__ == "__main__":
    # TODO: Test the training function and the loading function
    # TODO: Test the encode and decode function
    pass