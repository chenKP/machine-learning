import os.path

import numpy
import torch
import torchvision
from torch import  nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchvision import transforms
#定义显示图片的方法
from torchvision.utils import make_grid


def show_picture(inp, title=None):
    plt.figure(figsize=(14, 4))
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.486])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp +mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

#更改预训练模型的池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self,size = None):
        super().__init__()
        size = size or (1,1)
        self.pool_one = nn.AdaptiveAvgPool2d(size)
        self.pool_two = nn.AdaptiveAvgPool2d(size)

    def forward(self,x):
        return torch.cat([self.pool_one(x), self.pool_two(x), 1])#连接两个池化层


#迁移学习：用一个成熟的模型，进行微调
def get_model():
    model_pre = models.resnet50(pretrained=True)#获取预训练模型
    #冻结模型中的参数
    for param in model_pre.parameters():
        param.requires_grad = False
    #调整模型，替换最后两层网络，返回一个新模型
    #model_pre.avgpool = AdaptiveConcatPool2d()#池化层替换
    model_pre.fc = nn.Sequential(
        nn.Flatten(),#所有维度拉平
        nn.BatchNorm1d(2048),#正则化
        nn.Dropout(0.5),#丢掉一些神经元
        nn.Linear(2048,512),#线性层的处理
        nn.ReLU(),#激活函数
        nn.BatchNorm1d(512),
        nn.Linear(512,2),#线性，输入512，输出2
        nn.LogSoftmax(dim=1),#损失函数

    )
    return model_pre


#定义训练函数
def train(model, device, train_data, optimzer, epoch, writer,criterion):
    model.train()
    total_loss = 0
    for data,target in train_data:
        data ,target = data.to(device), target.to(device)
        optimzer.zero_grad#梯度初始化为0
        output = model(data)
        loss = criterion(output,target)#计算损失
        loss.backward()#反向传播
        optimzer.step()
        total_loss = total_loss + loss
    writer.add_scalar("Train loss",total_loss/len(train_data),epoch)
    print("train_loss: {}".format(total_loss/len(train_data)))
    writer.flush()#刷新
    return total_loss/len(train_data)#返回平均损失

#定义测试函数
def test(model, device, test_data, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data,target in test_data:
            data,target = data.to(device),target.to(device)
            #输出
            output = model(data)
            #计算损失
            total_loss = total_loss + criterion(output,target).item()
            #获取预测结果每行数据概率最大值的下标
            pread = output.max(1, keepdim=True)[1]
            #正确情况统计
            accuracy += pread.eq(target.view_as(pread)).sum().item()

        writer.add_scalar("Test_loss",total_loss/len(test_data),epoch)
        writer.add_scalar("Accuracy",accuracy/len(test_data),epoch)
        writer.flush()
        print("test_loss: {} accuracy: {}".format(total_loss/len(test_data),
                                                  accuracy/len(test_data)))
        return total_loss/len(test_data), accuracy/len(test_data)

def main():
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda")
    EPOCHS = 10
    writer = SummaryWriter("logs")
    LEARN_RATE = 0.01
    #图片转换
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        ])
    }

    #操作数据集
    #数据集路径
    data_path = "./chest_xray"
    #加载数据集
    image_data = {x : torchvision.datasets.ImageFolder(os.path.join(data_path,x),
                  data_transforms[x])
                  for x in ['train','val']}
    #读取数据
    data_loaders = {x : DataLoader(image_data[x],shuffle=True,batch_size=BATCH_SIZE)
                    for x in ["train", "val"]}
    #训练和验证数据集的大小
    data_size = {x : len(image_data[x])
                 for x in ["train","val"]}
    #获取标签的类别名称
    target_name = image_data["train"].classes

    #显示一个batch_size的图像
    data,target = next(iter(data_loaders["train"]))
    out = make_grid(data,nrow=8,padding=10)
    show_picture(out,title=[target_name[x]
                            for x in target])
    model = get_model()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(),lr=LEARN_RATE)
    for epoch in range(EPOCHS):
        print("---------------第 {} 次训练-------------".format(epoch))
        train(model,DEVICE,data_loaders["train"],optimzer,epoch,writer,criterion)
        test(model,DEVICE,data_loaders["val"],criterion,epoch,writer)

if __name__ == '__main__':
    main()