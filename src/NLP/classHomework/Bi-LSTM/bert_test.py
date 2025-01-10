from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json


class corpus(Dataset):
    def __init__(self, path, max_length=50):
        self.root = path
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.idx2label = []
        self.label2idx = {}
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1
        self.train = self.load_data('train.json')
        self.valid = self.load_data('dev.json')
        self.test = self.load_data('test.json', test_mode=True)

    def load_data(self, path, test_mode=False):
        bert_model = BertModel.from_pretrained('bert-base-chinese').to(device="mps")
        json_path = os.path.join(self.root, path)
        sentence_list = []
        label_list = []
        with open(json_path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                encoding = self.tokenizer(sent, truncation=True, padding=True, max_length=self.max_length, add_special_tokens=True, return_tensors="pt")
                for k, v in encoding.items():
                    encoding[k] = v.to(device="mps")
                with torch.no_grad():
                    encoding = bert_model(**encoding)
                sentence_list.append(encoding.last_hidden_state[0, 0].view(1, -1))
                if test_mode:
                    label = json.loads(line)['id']      
                    label_list.append(label)
                else:
                    label = json.loads(line)['label']
                    label_list.append(self.label2idx[label])

        sentence_list = torch.stack(sentence_list).to(device="cpu")
        label_list = torch.tensor(label_list).to(device="cpu")
        torch.save((sentence_list, label_list), path + ".pt")
        return bert_dataset(sentence_list, label_list)
    
class bert_dataset(Dataset):
    def __init__(self, sentence_list, label_list):
        self.sentence_list = sentence_list.squeeze(1)
        self.label_list = label_list

    def __getitem__(self, index):
        return self.sentence_list[index], self.label_list[index]

    def __len__(self):
        return len(self.sentence_list)



class bert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(768, 15)

    def forward(self, x):
        with torch.no_grad():
            x = self.bert(**x)

        x = self.fc(x.pooler_output)
        return x

def valid(one_token=False):
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))

if __name__ == "__main__":
    dataset_folder = "testDataset/tnews_public"
    output_folder = "experiment/output/Bi-LSTM_output"
    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 768     # 每个词向量的维度
    max_token_per_sent = 50 # 每个句子预设的最大 token 数
    batch_size = 128
    num_epochs = 20
    lr = 1e-4
    from_pretrained = False
    layer_num = 2
    n_head = 12
    one_token = False
    if from_pretrained:
        embedding_dim = 300
    #------------------------------------------------------end------------------------------------------------------#
    device = "mps"
    train_data, train_label = torch.load('train.json.pt')
    valid_data, valid_label = torch.load('dev.json.pt')
    test_data, test_label = torch.load('test.json.pt')
    train_dataset = TensorDataset(train_data, train_label)
    valid_dataset = TensorDataset(valid_data, valid_label)
    test_dataset = TensorDataset(test_data, test_label)
    model = nn.Sequential(nn.Linear(768, 15), nn.Softmax(dim=-1))
    model = model.to(device="mps")
    data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    max_valid_acc = 0
    summaryWriter = SummaryWriter(log_dir="runs/{}-{}-{}-{}-{}-{}".format(model._get_name(), layer_num, batch_size, num_epochs, one_token, from_pretrained))
    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # 模型预测
            y_hat = model(batch_x)

            loss = loss_function(y_hat.view(y_hat.shape[0], y_hat.shape[-1]), batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())
            summaryWriter.add_scalar("train_loss", loss.item(), global_step=epoch * len(data_loader_train) + len(total_loss))
            summaryWriter.add_scalar("train_acc", sum(total_true) / (batch_size * len(total_true)), global_step=epoch * len(data_loader_train) + len(total_loss))
            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid(one_token)

        summaryWriter.add_scalar("valid_acc", valid_acc, global_step=epoch)
        if valid_acc > max_valid_acc:
            torch.save(model, os.path.join(output_folder, "{}_model.ckpt".format(model._get_name())))
        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")
    summaryWriter.close()
    