# github@OpenGVLab,repo:LLaMA-Adapter, https://github.com/OpenGVLab/LLaMA-Adapter.git

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
from ...utils.DataLoader.levir_CC import levirDataset

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class CPUGradScaler:
    """
    如果没有可用的 GPU，就用这个“模拟”的 GradScaler，
    保证代码在 CPU 上运行时不报错。
    """
    def scale(self, loss):
        # 在 CPU 上不进行缩放，直接返回 loss
        return loss

    def step(self, optimizer):
        # 直接进行优化器的 step
        optimizer.step()

    def update(self):
        # CPU 上不需要真正地更新缩放因子，空实现即可
        pass

    def unscale_(self, optimizer):
        # CPU 上不需要进行 unscale，空实现
        pass

    def state_dict(self):
        # 返回空 dict
        return {}

    def load_state_dict(self, state_dict):
        # 不做任何处理
        pass

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        if torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = CPUGradScaler()
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def train_one_epoch(model: LlamaAdapter, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device, epoch: int, warmup_epochs: int, lr, min_lr, epochs,
                    accum_iter=64, 
                    summary_writer=None,):
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device=device)
    loss_scaler = NativeScaler()
    for data_iter_step, (examples, labels, example_mask, imgs) in tqdm(enumerate(data_loader), total=len(data_loader)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, 
                                 warmup_epochs=warmup_epochs, lr=lr, min_lr=min_lr, epochs=epochs)
        imgs = imgs.to(device=device)
        examples = examples.to(device=device)
        labels = labels.to(device=device)

        model_output = model(examples, imgs)
        loss = criterion(model_output.view(-1, model_output.shape[-1]), labels[:, 1:].view(-1))
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            break

        if summary_writer is not None:
            summary_writer.add_scalar("loss", loss.item(), data_iter_step)
        


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def train(config_path,
          max_seq_len=128, max_batch_size=1, phase="pretrain", 
          device="mps", clip_model='ViT-L/14'):
    config = INIconfig(config_path)
    model = LlamaAdapter(config.LLAMA['llama_ckpt_dir'], config.LLAMA['llama_tokenizer'], 
                         max_seq_len=max_seq_len, max_batch_size=max_batch_size, phase=phase, device=device, 
                         clip_model=clip_model)
    model.to(device=device)
    training_dataset = levirDataset(config.DATASET['levir_json'], config.DATASET['levir_img'], 
                                    max_len=max_seq_len, llama_token_dir=config.LLAMA['llama_tokenizer'], clip_model=clip_model)
    training_dataloader = DataLoader(training_dataset, batch_size=max_batch_size, shuffle=True, num_workers=8)
    param_groups = add_weight_decay(model, weight_decay=float(config.TRAIN['weight_decay']))
    optimizer = torch.optim.AdamW(param_groups, lr=float(config.TRAIN['lr']))
    summary_writer = SummaryWriter()

    for epoch in range(int(config.TRAIN['epochs'])):
        train_one_epoch(model, training_dataloader, optimizer, device, epoch, 
                        warmup_epochs=int(config.TRAIN['warmup_epochs']), lr=float(config.TRAIN['lr']), 
                        min_lr=float(config.TRAIN['min_lr']), epochs=int(config.TRAIN['epochs']), 
                        summary_writer=summary_writer)
        break
    torch.save(model.state_dict(), "experiment/output/llama_adapter/adapter_param.pt")