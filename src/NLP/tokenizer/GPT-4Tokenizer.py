import torch
import regex as re
from tqdm import tqdm

class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        

        # using regex to process the input text like gpt-4 do
        self.special_tokens = {
            "<|endoftext|>": 50257
        }

    def get_stats(self, ids):
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    def merge(self, ids, pair, idx):
        newids = []
        if len(ids) >= 2:
            i = 0
            while i < len(ids):
                try:
                    j = ids[i:].index(pair[0])
                except:
                    newids.extend(ids[i:])
                    break
                newids.extend(ids[i: j])
                i = j
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1

            return newids

        else:
            return ids
        

    def train(self, text, vocab_size, verbose=False):
        tokens = list(map(int, text.encode('utf-8')))
        with tqdm(total=vocab_size - 256) as pbar:
            while len(self.merges) < vocab_size - 256 - len(self.special_tokens):
                stats = self.get_stats(tokens)
                top_token = max(stats, key=stats.get)
                idx = 256 + len(self.merges)
                tokens = self.merge(tokens, top_token, idx)
                self.merges[top_token] = idx
                self.vocab[idx] = self.vocab[top_token[0]] + self.vocab[top_token[1]]
                if verbose:
                    print(f"merging {top_token} into {idx}")
                pbar.update(1)
            
            for j in range(len(self.special_tokens)):
                self.merges[self.special_tokens[list(self.special_tokens.keys())[j]]] = vocab_size - 256 - len(self.special_tokens) + j
                self.vocab[vocab_size - 256 - len(self.special_tokens) + j] = list(self.special_tokens.keys())[j].encode('utf-8')
                pbar.update(1)


    def encode(self, text):
        

        tokens = list(map(int, text.encode('utf-8')))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
            

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors='replace')

    def from_pretrained(self, urls):
        # TODO: Add from_pretrained function, loading merges and vocabe
        pass


if __name__ == "__main__":
    with open("/Users/a1/PythonProgram/zmcode/src/tokenizer/test/taylorswift.txt", encoding='utf-8') as f:
        texts = f.read()
    tokenizer = BasicTokenizer()
    tokenizer.train(texts, vocab_size=50257)