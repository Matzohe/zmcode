import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MultiModal.clip import tokenize
import src.MultiModal.clip as clip
from src.utils.DataLoader.book_corpus import BookCorpusDataset
from src.utils.utils import INIconfig, set_seed
from src.utils.DataLoader.book_corpus import get_book_corpus_dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class rnn_block(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(rnn_block, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 初始化权重
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.ouptut_linear = nn.Linear(hidden_size, output_size)

    def forward(self, data_input):
        x, h = data_input
        hidden_state = torch.tanh(self.input_linear(x) + self.hidden_linear(h))
        y = self.ouptut_linear(hidden_state)
        return y, hidden_state
    

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.input_size = int(config.RNN['input_size'])
        self.hidden_size = int(config.RNN['hidden_size'])
        self.output_size = int(config.RNN['output_size'])
        self.layers = int(config.RNN['layers'])
        self.device = config.TRAINING['device']
        state_dict = torch.load("embedding_weight.pt")
        self.token_embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.token_embedding.load_state_dict(state_dict)
        self.token_embedding.requires_grad_(False)
        self.predict = nn.Linear(self.hidden_size, self.output_size, bias=False)
        # share the weight
        self.predict.weight = self.token_embedding.weight
        self.predict.weight.requires_grad_(False)

        self.model = nn.ModuleList([
            rnn_block(self.hidden_size, self.hidden_size, self.hidden_size) for _ in range(self.layers)])
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.token_embedding(x)
        h = torch.zeros(size=(x.shape[0], self.layers, self.hidden_size)).to(self.device)

        assert len(x.shape) == 3
        output_words = []
        saved_tensor = []
        saved_tensor.append(h)
        for l in range(x.shape[1]):

            if self.layers == 1:
                output, output_h = self.model((x[:, l, :].view(x.shape[0], -1), saved_tensor[-1][:, 0, :].view(h.shape[0], -1)))
                saved_tensor.append(output_h.unsqueeze(1))
                output_words.append(output.unsqueeze(1))
            
            else:
                h_list = []
                for i, layer in enumerate(self.model):
                    if i == 0:
                        output, output_h = layer((x[:, l, :].view(x.shape[0], -1), saved_tensor[-1][:, i, :].view(h.shape[0], -1)))
                        h_list.append(output_h.unsqueeze(1))
                    else:
                        output, output_h = layer((output, saved_tensor[-1][:, i, :].view(h.shape[0], -1)))
                        h_list.append(output_h.unsqueeze(1))
                saved_tensor.append(torch.cat(h_list, dim=1))
                output_words.append(output.unsqueeze(1))
        
        output_words = torch.cat(output_words, dim=1)
        output_words = torch.softmax(self.predict(output_words), dim=-1)
        return output_words

    def generate(self, x, generated_length=128):
        
        x = self.token_embedding(x)
        h = torch.zeros(size=(1, 1, self.hidden_size)).to(self.device)
        assert len(x.shape) == 2
        saved_tensor = []
        saved_tensor.append(h)
        output_words = []
        for l in range(x.shape[0]):
            
            if self.layers == 1:
                output, output_h = self.model(x[l, :], saved_tensor[-1][:, 0, :])
                saved_tensor.append(output_h)
                output_words.append(output)
            
            else:
                for i, layer in enumerate(self.model):
                    output, h[:, i, :] = layer(x[:, l, :], h[:, i, :])        

        for j in range(generated_length):

            if self.layers == 1:
                output, h[:, 0, :] = self.model((output_words[-1], h[:, 0, :]))
                output_words.append(output)
            
            else:
                for i, layer in enumerate(self.model):
                    if i == 0:
                        output, h[:, i, :] = layer((output_words[-1], h[:, i, :]))
                    else:
                        output, h[:, i, :] = layer((output, h[:, i, :]))
                    
                output_words.append(output)
        
        output_words = torch.cat(output_words, dim=0)
        output_words = torch.softmax(self.predict(output_words), dim=-1)

        return output_words


def rnn_training():
    summary_writer = SummaryWriter()
    set_seed(42)
    config = INIconfig("config/rnn_config.cfg")
    model = RNN(config)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    dataloader = get_book_corpus_dataloader(config)

    model = model.to(config.TRAINING['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.TRAINING['lr']), weight_decay=float(config.TRAINING['weight_decay']))

    for epoch in range(int(config.TRAINING['epoch'])):

        for idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.to(config.TRAINING['device'])
            y = y.to(config.TRAINING['device'])

            output = model(x)
            loss = loss_function(output.permute(0, 2, 1), y)

            summary_writer.add_scalar("loss", loss.detach().cpu(), idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    
