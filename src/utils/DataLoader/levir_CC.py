from ...MultiModal.llama_adapter.llama.tokenizer import Tokenizer
from ...MultiModal import clip
import torch
from torch.utils.data import DataLoader, Dataset
import random
import json
import os
import copy
import cv2
from PIL import Image


def format_prompt(instruction, input=None):

    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})



class levirDataset(Dataset):
    def __init__(self, json_root, image_root, max_len, llama_token_dir, clip_model):
        clip_model, self.img_preprocess = clip.load(clip_model, device="cpu")
        del clip_model

        with open(json_root, "r") as f:
            self.data = json.load(f)["images"]
        self.image_path_list = []
        self.image_discription = {}
        self.max_len = max_len
        self.tokenizer = Tokenizer(llama_token_dir)
        for each in self.data:
            if each["split"] != "train":
                continue
            img_path = os.path.join(image_root, each["filepath"], "A", each["filename"])
            self.image_path_list.append(img_path)
            sentence = each["sentences"]
            sentence_list = []
            for sent in sentence:
                sentence_list.append(sent["raw"][1:])
            self.image_discription[img_path] = sentence_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.img_preprocess(img)
        format_instruction = "Generate caption of this image"
        input1 = format_prompt(format_instruction, None)
        sentence = self.image_discription[img_path]
        if len(sentence) > 1:
            sentence = random.choice(sentence)
        else:
            sentence = sentence[0]
        input2 = input1 + sentence
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_len - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_len]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, img
