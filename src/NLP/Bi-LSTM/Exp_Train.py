import torch
import torch.nn as nn
import time
import json
import os
from bert_test import corpus, bert
from transformers import BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus
from Exp_Model import BiLSTM_model, Transformer_model
from transformers import TrainingArguments, Trainer


def train(layer_num, batch_size, epoch_num, one_token, from_pretrained):
    '''
    进行训练
    '''
    max_valid_acc = 0
    summaryWriter = SummaryWriter(log_dir="runs/{}-{}-{}-{}-{}-{}".format(model._get_name(), layer_num, batch_size, epoch_num, one_token, from_pretrained))
    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # 模型预测
            y_hat = model(batch_x, one_token=one_token)

            loss = loss_function(y_hat, batch_y)

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


def valid(one_token=False):
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x, one_token=one_token)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def predict(one_token=False):
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt")).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            batch_x, batch_y = data[0].to(device), data[1]

            y_hat = model(batch_x, one_token=one_token)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            test_ids += batch_y.tolist()
            test_pred += y_hat.tolist()

    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w") as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.dictionary.idx2label[label_idx][1]
            json_data = json.dumps(one_data)    # 将字典转为json格式的字符串
            f.write(json_data + "\n")
            

if __name__ == '__main__':
    dataset_folder = "testDataset/tnews_public"
    output_folder = "experiment/output/Bi-LSTM_output"

    device = "mps"

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

    dataset = Corpus(dataset_folder, max_token_per_sent, from_pretrained=from_pretrained)

    print("finished loading dataset")

    vocab_size = len(dataset.dictionary.tkn2word)
    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 可修改选择的模型以及传入的参数
    # if from_pretrained:
    #     embedding_weight = dataset.embedding_weight
    #     model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim, embedding_weight=embedding_weight).to(device)      
    # else:
    #     model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim, nlayers=layer_num).to(device)   
    if from_pretrained:
        embedding_weight = dataset.embedding_weight
        model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, nhead=n_head, d_emb=embedding_dim, embedding_weight=embedding_weight).to(device)
    else:
        model = Transformer_model(vocab_size=vocab_size, ntoken=max_token_per_sent, nhead=n_head, d_emb=embedding_dim, nlayers=layer_num).to(device)          


    # #------------------------------------------------------end------------------------------------------------------#
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  
    






    # # 进行训练
    # bert_train(layer_num, batch_size, num_epochs, one_token, from_pretrained)

    # # 对测试集进行预测
    # predict(one_token=one_token)
