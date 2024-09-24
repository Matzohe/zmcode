from torch_cka import CKA
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
 


def cka_test():
    # 加载预训练模型
    model1 = models.resnet18(pretrained=True)
    model2 = models.resnet34(pretrained=True)

    # 准备数据集
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='testDataset/cifar', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 初始化 CKA 比较工具
    cka = CKA(model1, model2, model1_name="ResNet18", model2_name="ResNet34", device='cuda')
    
    # 进行比较
    cka.compare(dataloader)
    
    # 导出结果
    results = cka.export()
    print(results)