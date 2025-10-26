import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Linear, ReLU
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

 
 
# Dataset:创建数据集的函数；__init__:初始化数据内容和标签
# __geyitem:获取数据内容和标签
# __len__:获取数据集大小
# daataloader:数据加载类，接受来自dataset已经加载好的数据集
# torchbision:图形库，包含预训练模型，加载数据的函数、图片变换，裁剪、旋转等
# torchtext:处理文本的工具包，将不同类型的额文件转换为datasets
 
# 预处理：将两个步骤整合在一起
transform = transforms.Compose({
    transforms.ToTensor(),  # 将灰度图片像素值（0~255）转为Tensor（0~1），方便后续处理
    # transforms.Normalize((0.1307,),(0.3081)),    # 归一化，均值0，方差1;mean:各通道的均值std：各通道的标准差inplace：是否原地操作
})
 
# normalize执行以下操作：image=(image-mean)/std?????
# input[channel] = (input[channel] - mean[channel]) / std[channel]
 
# 加载数据集
# 训练数据集
'''
介绍一下mnist数据集：
MNIST数据集一共有7万张图片，其中6万张是训练集，1万张是测试集。每张图片是28 × 28 28\times 2828×28的0 − 9 0-90−9的手写数字图片组成。每个图片是黑底白字的形式，黑底用0表示，白字用0-1之间的浮点数表示，越接近1，颜色越白。
MNIST数据集下载地址是http://yann.lecun.com/exdb/mnist/
参考：https://zhuanlan.zhihu.com/p/155748813
'''
# 注意：这个MNIST是官方给的一个写好的，后面可能会要求你们自己写这个类哦(‵﹏′)->作为某次作业的一部分
'''
这个类继承了torch.utils.data的Dataset类
有这么几个方法：
1. __init__()：初始化
2.__get_item__()：获取单个样本：返回图像和标签的张量
3.__len__():返回数据集样本数量


这里是一个示例：
class MNISTDataset(Dataset):
    def __init__(self, root, train=True):
        """初始化：加载数据"""
        self.root = root
        self.train = train
        
        # 选择数据文件
        if self.train:
            img_file = 'train-images-idx3-ubyte.gz'
            lbl_file = 'train-labels-idx1-ubyte.gz'
        else:
            img_file = 't10k-images-idx3-ubyte.gz'
            lbl_file = 't10k-labels-idx1-ubyte.gz'
        
        # 加载并预处理数据（将数据加载逻辑直接放在__init__中）
        with gzip.open(os.path.join(root, img_file), 'rb') as f:
            # 读取图像：跳过16字节头部，剩余为像素数据
            self.images = np.frombuffer(f.read(), np.uint8, offset=16)
            self.images = self.images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        
        with gzip.open(os.path.join(root, lbl_file), 'rb') as f:
            # 读取标签：跳过8字节头部，剩余为标签数据
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
    def __getitem__(self, idx):
        """获取单个样本：返回图像和标签的张量"""
        # 转换为PyTorch张量
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    def __len__(self):
        """返回样本总数"""
        return len(self.images)
'''
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# transform：指示加载的数据集应用的数据预处理的规则，shuffle：洗牌，是否打乱输入数据顺序
# 测试数据集
test_data = MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
 
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))
# print(test_data)
# print(train_data)
 
#这是对mnist进行识别的模型定义部分
'''
1.继承了torch的nn.module模型
下面是一个CNN模型，其有以下几层：
输入层（Input layer）
卷积层（convolutional layer）：提取有效信息
池化层（pooling layer）
输出层（全连接层＋softmax layer）
理论部分可以看:https://zhuanlan.zhihu.com/p/104776627
              https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html
'''
class MnistModel(nn.Module):
    def __init__(self):
        #这句的意思时继承父类，它的作用是调用当前类（MnistModel）的父类（通常是 nn.Module）的构造函数，确保父类能够正确初始化。其中super(MnistModel, self)为获取其父类的函数
        super(MnistModel, self).__init__()
       '''
        conv2d:2维卷积
        in_channels=1：输入特征图的通道数为 1（通常用于灰度图像）
        out_channels=10：输出特征图的通道数为 10（表示该卷积层有 10 个不同的卷积核）
        kernel_size=5：卷积核的大小为 5x5
        stride=1：卷积步长为 1（每次滑动 1 个像素）
        padding=0：不使用填充（卷积后特征图尺寸会缩小）
       '''
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        '''
        这行代码在 PyTorch 中定义了一个二维最大池化层（MaxPool2d），用于对卷积后的特征图进行下采样。
        2 表示池化窗口的大小为 2x2（这是最常见的设置）
        该层会在输入特征图上滑动 2x2 的窗口，取每个窗口中的最大值作为输出
        '''
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = MaxPool2d(2)
        '''
        下面定义了三个全连接层
        具体参数解释：
        320：输入特征的维度（即输入该层的张量最后一个维度的大小）
        128：输出特征的维度（该层会将输入特征映射到 128 维的新特征空间）
        全连接层的数学表达为：y = xW + b，其中：
        x 是输入张量（形状为 [batch_size, 320]）
        W 是该层的权重矩阵（形状为 [320, 128]）
        b 是偏置向量（形状为 [128]）
        y 是输出张量（形状为 [batch_size, 128]）
        '''
        self.linear1 = Linear(320, 128)
        #输入维度必须与前一层输出维度匹配
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        #激活函数
        self.relu = ReLU()
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu(x)
        #如果线性层前一层是卷积 / 池化层，需要先通过 view() 或 flatten() 将特征图展平为一维向量，此时 in_features 等于展平后的向量长度。
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
 
        return x
 
 
# 损失函数CrossentropyLoss
model = MnistModel()#实例化
criterion = nn.CrossEntropyLoss()   # 交叉熵损失
# SGD，优化器，梯度下降算法
optimizer = torch.optim.SGD(model.parameters(), lr=0.14)#lr:学习率
# 模型训练
def train():
    # index = 0
    for index, data in enumerate(train_loader):#获取训练数据以及对应标签
        # for data in train_loader:
       input, target = data   # input为输入数据，target为标签
       y_predict = model(input) #模型预测
       loss = criterion(y_predict, target)
       optimizer.zero_grad() #梯度清零
       loss.backward()#loss值反向传播
       optimizer.step()#更新参数
       # index += 1
       if index % 100 == 0: # 每一百次保存一次模型，打印损失
           torch.save(model.state_dict(), "./model/model.pkl")   # 保存模型
           torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
           print("训练次数为：{}，损失值为：{}".format(index, loss.item() ))
# 模型测试
def test():
    correct = 0     # 正确预测的个数
    total = 0   # 总数
    with torch.no_grad():   # 测试不用计算梯度
        for data in test_loader:
            input, target = data
            output = model(input)   # output输出10个预测取值，概率最大的为预测数
            probability, predict = torch.max(input=output.data, dim=1)    # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
            # loss = criterion(output, target)
            total += target.size(0)  # target是形状为（batch_size,1)的矩阵，使用size（0）取出该批的大小
            correct += (predict == target).sum().item()  # predict 和target均为（batch_size,1)的矩阵，sum求出相等的个数 item()转换为数字
        print("测试准确率为：%.6f" %(correct / total))
 
 
#测试识别函数
if __name__ == '__main__':
    #训练与测试
    for i in range(15):#训练和测试进行5轮
        print({"————————第{}轮测试开始——————".format (i + 1)})
        train()
        test()