import random
import os
from tqdm import tqdm

def emotional_data_preprocess(config):
    neg_path = os.path.join(config.DATASET['hotel_emotion'], "negative")
    pos_path = os.path.join(config.DATASET['hotel_emotion'], "positive")
    train_path = os.path.join(config.DATASET['hotel_emotion'], "train.txt")
    test_path = os.path.join(config.DATASET['hotel_emotion'], "test.txt")

    # seperate of the training set
    train_split = float(config.HOTEL['train_split'])

    files_pos = os.listdir(pos_path)
    random.shuffle(files_pos)
    files_neg = os.listdir(neg_path)
    random.shuffle(files_neg)
    
    # generate training dataset
    with open(train_path, 'w') as tar:
        for file in tqdm(files_pos[0:int(train_split * len(files_pos))], desc="generate training possitive dataset"):
            path = os.path.join(pos_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = ""
                for line in f:
                    if line[:-1] == "":
                        continue
                    content += line[:-1]
                    content += '。'
                content = "1"+content+'\n'
                tar.write(content)
        for file in tqdm(files_neg[0:int(train_split*len(files_pos))], desc="generate training negative dataset"):
            path = os.path.join(neg_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = ""
                for line in f:
                    if line[:-1] == "":
                        continue
                    content += line[:-1]
                    content += '。'
                content = "0" + content + '\n'
                tar.write(content)
    
    # generate testing dataset
    with open(test_path, 'w') as tar:
        for file in tqdm(files_pos[int(train_split*len(files_pos)):], desc="generate testing possitive dataset"):
            path = os.path.join(pos_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = ""
                for line in f:
                    if line[:-1] == "":
                        continue
                    content += line[:-1]
                    content += '。'
                content = "1"+content+'\n'
                tar.write(content)
        for file in tqdm(files_neg[int(train_split * len(files_pos)):], desc="generate testing negative dataset"):
            path = os.path.join(neg_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = ""
                for line in f:
                    if line[:-1] == "":
                        continue
                    content += line[:-1]
                    content += '。'
                content = "0" + content + '\n'
                tar.write(content)
