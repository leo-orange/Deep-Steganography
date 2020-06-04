import torch
import torch.nn as nn
'''
nn.Conv2d()
in_channels：输入信号的通道数
out_channels：卷积产生的通道数
kernel_size：卷积核大小
stride：步长大小
padding：补0
dilation：kernel间距
'''

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
# 隐藏图像网络
class Hide(nn.Module):
    def __init__(self):
        super(Hide, self).__init__()
        # 作为一个容器，模块将按照构造函数中传递的顺序添加到模块中
        self.prepare = nn.Sequential(
            # 输入3维度、输出64维度、卷积核大小3、步长1、补1
            conv3x3(3, 64),
            # 激活函数
            # inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.ReLU(True),
            # 输入64维度、输出64维度、步长为2、补1
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.hidding_1 = nn.Sequential(
            # 输入128维度、输出64维度、卷积核大小1、步长2、补0
            nn.Conv2d(128, 64, 1, 1, 0),
            # 数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 输入64维度、输出64维度、卷积核大小3、步长1、补1
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 输入64维度、输出32维度、卷积核大小3、步长1、补1
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        #
        self.hidding_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 输入32维度、输出16维度、卷积核大小3、步长1、补1
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 输入16维度、输出3维度、卷积核大小3、步长1、补1
            conv3x3(16, 3),
            # 激活函数
            nn.Tanh()
        )
    def forward(self, secret, cover):
        # 将秘密图像进行处理

        sec_feature = self.prepare(secret)
        # 对载体图像进行处理
        cover_feature = self.prepare(cover)

        # 将训练好的两张图片横向拼接，再进行训练
        out = self.hidding_1(torch.cat([sec_feature, cover_feature], dim=1))

        out = self.hidding_2(out)
        return out
# 显示秘密图像网络
class Reveal(nn.Module):
    def __init__(self):
        super(Reveal, self).__init__()
        self.reveal = nn.Sequential(
            conv3x3(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            conv3x3(16, 3),
            nn.Tanh()
        )

    def forward(self, image):
        out = self.reveal(image)
        return out

