from transformers import AutoTokenizer, AutoModel
import torch
import csv
from tqdm import tqdm
import json


img_discription_list = []
with open("/Volumes/T7/zmcode/testDataset/Levir-CC-dataset/LevirCCcaptions.json", "r") as f:
    data = json.load(f)
    data = data['images']
    for each in data:
        if each["split"] != "test":
            continue
        sentence = each["sentences"]
        sentence_list = []
        for sent in sentence:
            sentence_list.append(sent["raw"][1:])
        img_discription_list.append(sentence_list)
    
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
all_similarity_number = 0

with torch.no_grad():
    for i in tqdm(range(len(img_discription_list)), total=len(img_discription_list)):
        img_discription = img_discription_list[i]
        if len(img_discription) == 1:
            all_similarity_number += 1 / len(img_discription_list)
        embedding_list = []
        for each in img_discription:
            original_inputs = tokenizer(each, padding=True, truncation=True, return_tensors="pt")
            original_outputs = model(**original_inputs)
            original_embedding = original_outputs.last_hidden_state[:, 0, :].view(-1)
            embedding_list.append(original_embedding)
        similarity_sum = 0
        for j in range(len(embedding_list) - 1):
            similarity_sum += torch.nn.functional.cosine_similarity(embedding_list[j], embedding_list[j + 1], dim=0)
        all_similarity_number += similarity_sum / (len(img_discription_list) * (len(embedding_list) - 1))
        
print(all_similarity_number)

        

