import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class FakeNewsBertClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.BERTCLASSIFICATION['device']
        self.model = BertModel.from_pretrained('bert-base-chinese').eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.unfrozen_list = eval(config.BERTCLASSIFICATION['unfrozen_list'])
        self.classification = nn.Linear(3 * self.model.embeddings.word_embeddings.embedding_dim, int(config.BERTCLASSIFICATION['num_class']))
        self.softmax = nn.Softmax(dim=-1)

        for name, param in self.model.named_parameters():
            if any(word in name for word in self.unfrozen_list):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, texts):
        final_output = []
        info_input = [[], [], []]
        for j in range(len(texts)):
            for i, text in enumerate(texts[j]):
                info_input[i % 3].append(text)

        for i in range(len(info_input)):
            inputs= self.tokenizer(info_input[i], return_tensors="pt", padding=True, truncation=True)
            output = self.model(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device), inputs['token_type_ids'].to(self.device))
            output = output.last_hidden_state[:, 0, :]
            final_output.append(output)

        output = torch.cat(final_output, dim=-1)
        output = self.classification(output)
        output = self.softmax(output)

        return output