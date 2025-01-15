from transformers import AutoTokenizer, AutoModel
import torch
import csv
from tqdm import tqdm

img_path_list = []
img_discription_list = []
generated_discription_list = []
with open("/Users/a1/PythonProgram/zmcode/experiment/output/llama_adapter/generation.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for line in reader:
        img_path_list.append(line[2])
        img_discription_list.append(eval(line[0])[0])
        generated_discription_list.append(line[1])

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
all_similarity_number = 0
with torch.no_grad():
    for i in tqdm(range(0, 102), total=102):
        img_discription = img_discription_list[i]
        generated_discription = generated_discription_list[i]
        original_inputs = tokenizer(img_discription, padding=True, truncation=True, return_tensors="pt")
        generated_inputs = tokenizer(generated_discription, padding=True, truncation=True, return_tensors="pt")

        original_outputs = model(**original_inputs)
        generated_outputs = model(**generated_inputs)

        original_embedding = original_outputs.last_hidden_state[:, 0, :].view(-1)
        generated_embedding = generated_outputs.last_hidden_state[:, 0, :].view(-1)
        cosine_sim = torch.nn.functional.cosine_similarity(original_embedding, generated_embedding, dim=0)
        all_similarity_number += cosine_sim / 102
        

print(all_similarity_number)
