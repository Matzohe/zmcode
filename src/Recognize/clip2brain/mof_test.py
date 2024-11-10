output_dir = "/data/guoyuan/haofang/clip2brain/output" # 编码结果存储的路径
features_dir = "/data/guoyuan/haofang/clip2brain/features" # feature 存储的路径
model = "clip_vit"
# subj = 5
fix_testing = 42 # random seed
roi = "SELECTIVE_ROI"


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


import pickle
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

import torch
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

from src.encodingmodel.ridge import RidgeCVEstimator

from scipy.stats import pearsonr

from src.util.util import r2_score

from tqdm import tqdm

if __name__ == '__main__':

    subj = 5
    roi = "SELECTIVE_ROI"
    model = "clip_vit"

    # Load brain data
    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d%s.npy"
        % (output_dir, subj, roi)
    )

    br_data = np.load(brain_path)
    print("Brain data shape:",br_data.shape)

    trial_mask = np.sum(np.isnan(br_data), axis=1) <= 0
    br_data = br_data[trial_mask, :]

    print("NaNs? ",np.any(np.isnan(br_data)))
    print("Finite? ",np.all(np.isfinite(br_data)))

    stimulus_list = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (output_dir, subj)
    )

    # load feature matrix

    layer_modifier = ""
    featmat = np.load(
                "%s/subj%d/%s%s.npy" % (features_dir, subj, model, layer_modifier)
            )
      
    feature_mat = featmat.squeeze()
    feature_mat = feature_mat[trial_mask, :]


    # print("unsqueeze:",featmat.shape) # (10000, 1 ,512)
    # print("squeeze:", feature_mat.shape) # (10000, 512)

    fm = feature_mat
    br = br_data

    print("data to train")
    print("feature mat:", feature_mat.shape)
    print("real response:", br_data.shape)

    # whether set cross-validation True or False

    X, y = fm, br

    # shuffle the data in an fixed way

    random_seed = 42

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=random_seed,
        shuffle=True
    )


    X_train = torch.from_numpy(X_train).to(dtype=torch.float32).to(device)
    X_val = torch.from_numpy(X_val).to(dtype=torch.float32).to(device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float32).to(device)
    y_val = torch.from_numpy(y_val).to(dtype=torch.float32).to(device)
    
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegressionModel, self).__init__() # 这一行的作用？
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            out = self.linear(x)
            return out

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = LinearRegressionModel(input_dim, output_dim)

    # 使用均方误差损失函数
    criterion = nn.MSELoss()

    # 使用 Adam 优化器
    lr_init = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay = 1.5e-2)

    num_epochs = 50
    batch_size = 64

    from torch.utils.data import DataLoader, TensorDataset

    # 创建训练集的数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 创建验证集的数据集和数据加载器
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    decay_rate = 5e-1



    for epoch in range(num_epochs):
        new_lrate = lr_init * (decay_rate ** (epoch / num_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        
        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    model.eval()

    repeat = 2000
    seed = 41

    X_mean = X_train.mean(dim=0, keepdim=True)

    np.random.seed(seed)
    label_idx = np.arange(X_val.shape[0])

    
    rsq_dist = list()

    for _ in tqdm(range(repeat)):
        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
        X_val_sampled = X_val[sampled_idx, :]
        yhat = model(X_val_sampled - X_mean)
        y_val_sampled = y_val[sampled_idx, :]
        rsqs = r2_score(y_val_sampled.cpu().numpy(), yhat.cpu().detach().numpy())
        rsq_dist.append(rsqs)

    print("mean rsq", np.mean(rsq_dist))

    path = "/data/guoyuan/haofang/clip2brain/output/bootstrap/clip_vit_full_subj5/"
    if not os.path.isdir(path):
        os.makedirs(path)
    np.save(path + "rsq_dist.npy", rsq_dist)
    

