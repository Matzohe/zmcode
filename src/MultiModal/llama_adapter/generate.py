import math
import sys
from typing import Iterable
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import inf
from ...utils.utils import INIconfig
from .lr_sched import adjust_learning_rate
from .llama.llama_adapter import LlamaAdapter
from ...utils.DataLoader.levir_CC import levirDataset, levirTestDataset
import csv

def generate_captions(config_path, max_seq_len=128, 
                      max_batch_size=1, 
                      phase="inference", 
                      device="mps", 
                      max_generation_len = 128, 
                      clip_model='ViT-L/14'):
    config = INIconfig(config_path)
    model = LlamaAdapter(config.LLAMA['llama_ckpt_dir'], config.LLAMA['llama_tokenizer'], 
                         max_seq_len=max_seq_len, max_batch_size=max_batch_size, phase=phase, device=device, 
                         clip_model=clip_model)
    state_dict = torch.load("experiment/output/llama_adapter/adapter_param_2.pt")
    model.load_state_dict(state_dict)
    model.train(False)
    model.to(device)
    testDataset = levirTestDataset(config.DATASET['levir_json'], config.DATASET['levir_img'], 
                                   max_len=max_seq_len, llama_token_dir=config.LLAMA['llama_tokenizer'], clip_model=clip_model)

    test_dataloader = DataLoader(testDataset, batch_size=1, shuffle=False)
    max_generation_len = 128
    data = [
        ["pre discription", "generated discription", "img path"],
    ]

    for i, (examples, imgs, imgs_B, sentence, img_path) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        examples = examples.to(device=device)
        imgs = imgs.to(device=device)
        imgs_B = imgs_B.to(device=device)
        imgs = torch.cat((imgs, imgs_B), 1)
        decoding = model.generate(imgs, examples, max_generation_len)
        data.append([sentence, decoding[0], img_path])

    with open("experiment/output/llama_adapter/generation.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
    


