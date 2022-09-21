import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
BATCH_SIZE = 64
#处理图片
img_transforms ={
    "train":transforms.Compose([
    transforms.RandomResizedCrop(size=300,scale=(0.8,1.1)),#随机长宽裁剪图片
    transforms.RandomRotation(degrees=10),#依据degree随机旋转一定角度
    transforms.ColorJitter(0.4,0.4,0.4),#修改亮度，对比度，饱和度
    transforms.RandomHorizontalFlip(),#水平反转
    transforms.CenterCrop(size=256),#根据给定的size从中心裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
                            ]),
    "val":transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "test":transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    ])
}

#加载数据集
data_dir = "./chest_xray/"
train_dir = data_dir + "train/"
val_dir = data_dir + "val/"
test_dir = data_dir + "test/"
#从文件读取数据
datasets = {
    "train":torchvision.datasets.ImageFolder(train_dir, transform=img_transforms["train"]),
    "val":torchvision.datasets.ImageFolder(val_dir,transform=img_transforms["val"]),
    "test":torchvision.datasets.ImageFolder(test_dir,transform=img_transforms["test"])
}

dataloaders = {
    "train":DataLoader(datasets["train"],batch_size=BATCH_SIZE,shuffle=True,drop_last=True),
    "val":DataLoader(datasets["val"], batch_size=BATCH_SIZE, shuffle=True),
    "test":DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
}

LABEL = dict((v,k) for k,v in datasets["train"].class_to_idx.items())

dataloaders["train"].dataset.root   #这个方法可以获取train数据的路径

#正常图片路径列表
file_normal = os.listdir(os.path.join(str(dataloaders["train"].dataset.root),"NORMAL"))
#感染图片路径列表
file_pne = os.listdir(os.path.join(str(dataloaders["train"].dataset.root), "PNEUMONIA"))

log_path = "logs"
#定义函数，创建tensorboard writer
def tb_writer():
    #time_str = time.strftime("%Y%m%d_%H%M")#获取事件
    #writer = SummaryWriter(log_path+time_str)
    writer = SummaryWriter(log_path)
    return writer
#创建writer
writer = tb_writer()
#获取一张图片
img = dataloaders["train"].dataset[0]
writer.add_image("test_show",img[0],0)
#记录错误分类的图片
def failed_classfy(predict,target,writer,images,count,epoch):
    if predict.size(0) == target.size(0):

        failed_classes = (predict.data != target.data)
        for index,image in enumerate(images[failed_classes]):

            img_name = "EPOCH:{};Predict:{};Actual:{}".format(epoch,LABEL[predict[failed_classes].tolist()[index]],
                                                              LABEL[target[failed_classes].tolist()[index]])
            writer.add_image(img_name, image, epoch)
#自定义池化层
class AdaptiveConcatPooll2d(nn.Module):
    def __init__(self,size=None):
        super(AdaptiveConcatPooll2d, self).__init__()
        self.avgpooling = nn.AdaptiveAvgPool2d(size)
        self.maxpooling = nn.AdaptiveMaxPool2d(size)
    def forward(self,x):
        return torch.cat([self.maxpooling(x),self.avgpooling(x)],dim=1)

#迁移学习
def get_model():
    #获取训练模型
    model = models.resnet50(pretrained=True)
    #冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    #替换最后两层，池化层和全连接层
    model.avgpool = AdaptiveConcatPooll2d((1,1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),#加速神经网络收敛过程
        nn.Dropout(0.5),#丢掉部分神经元
        nn.Linear(4096,512),
        nn.ReLU(),#激活函数
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512,2),
        nn.LogSoftmax(dim=1)#损失函数，将input转换成概率分布的形式
    )

    return model

#定义训练函数
def model_train_val(model,train_data,val_data,optimzer,loss_fun,device,writer,epoch):
    model.train()
    train_loss = 0
    train_size = 0
    for images,labels in train_data:
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss = loss_fun(output,labels)
        train_loss +=loss.item()
        train_size +=images.size(0)
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
    print("-----Train Loss:{}------".format(train_loss/train_size))
    writer.add_scalar("Train Loss",train_loss/train_size,epoch)
    writer.flush()

    model.eval()
    with torch.no_grad():
        val_size = 0
        val_loss = 0
        correct = 0
        for images,labels in val_data:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = loss_fun(output,labels)
            val_loss += loss.item()*images.size(0)
            val_size += images.size(0)
            pred = torch.max(output,dim=1)[1]
            correct += (pred == labels).sum()
        print("---------Epoch:{};Val Loss:{},Accuracy:{}-----".format(epoch, val_loss/val_size, correct/val_size))
        writer.add_scalar("Val Loss",val_loss/val_size,epoch)#平均损失
        writer.add_scalar("Val Accuracy",correct/val_size,epoch)#平均准确率
        return train_loss/train_size,val_loss/val_size,correct/val_size

#定义测试函数
def test_model(model,test_data,epoch,device):
    model.eval()
    test_size = 0
    test_acc = 0
    corr = 0
    for images,labels in test_data:
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        pred = torch.max(output,dim=1)[1]
        corr += (pred == labels).sum()
        test_size += images.size(0)
        failed_classfy(pred,labels,writer,images,10,epoch)
    test_acc = corr/test_size
    print("Test accuracy : {}".format(test_acc))
    writer.add_scalar("Test Accuracy",test_acc,epoch)
    return  test_acc
LEARN_RATE = 0.01
DEVICE = torch.device("cuda")
model = get_model()
model = model.to(DEVICE)
loss_fun = nn.NLLLoss()
optimzer = torch.optim.SGD(model.parameters(),LEARN_RATE)

def train_steps(model,train_data,test_data,val_data,optimzer,loss_fun,device,steps,writer):
    tr_loss = 0
    val_loss = 0
    val_acc = 0
    test_acc = 0
    best_acc = 0
    for epoch in range(steps):
        tr_loss,val_loss,val_acc = model_train_val(model,train_data,val_data,optimzer,loss_fun,device,writer,epoch)
        test_acc = test_model(model,test_data,epoch,device)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),"model.pth")
    return

EPOCHS = 10
train_steps(model,dataloaders["train"],dataloaders["test"],
            dataloaders["val"],optimzer,loss_fun,DEVICE,EPOCHS,writer)
writer.close()

def plot_confusion(cm):
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8),cmap=plt.cm.Blues)
    plt.xticks(range(2),['Normal','Pneumonia]'],fontsize=14)
    plt.yticks(range(2),['Normal','Pneumonia]'],fontsize=14)
    plt.xlabel('Predict Label',fontsize=16)
    plt.ylabel('True Label',fontsize=16)
    plt.show()
def accuracy(output,labels):
    preds = torch.max(output,dim=1)[1]
    correct = torch.tensor(torch.sum(preds==labels).item())/len(preds)
    return correct

def metrics(output,labels):
    preds = torch.max(output,dim=1)[1]
    cm = confusion_matrix(labels.cpu().numpy(),preds.cpu().numpy())
    plot_confusion(cm)
    tn,fp,fn,tp = cm.ravel()
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2*((precision * recall)/(precision+recall))
    return precision,recall,f1
precisions = []
recalls = []
fls = []
accuracies = []
with torch.no_grad():
    model.eval()
    for data,label in dataloaders["test"]:
        data,label = data.to(DEVICE),label.to(DEVICE)
        output = model(data)
        precision,recall,f1 = metrics(output,label)
        acc = accuracy(output,label)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(acc.item())
        fls.append(f1)