import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes=10
        # VGG16 的 5 个卷积块
        # Block 1: [Conv-Conv-MaxPool]
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: [Conv-Conv-MaxPool]
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: [Conv-Conv-Conv-MaxPool]
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: [Conv-Conv-Conv-MaxPool]
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 5: [Conv-Conv-Conv-MaxPool]
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        # CIFAR-10 图像大小为 32x32，经过 5 次 MaxPool(stride=2) 后变为 1x1
        # 所以最后特征图大小为 512 * 1 * 1 = 512
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 32, 32] 对于 CIFAR-10
        
        # 特征提取
        x = self.block1(x)   # [B, 64, 16, 16]
        x = self.block2(x)   # [B, 128, 8, 8]
        x = self.block3(x)   # [B, 256, 4, 4]
        x = self.block4(x)   # [B, 512, 2, 2]
        x = self.block5(x)   # [B, 512, 1, 1]
        
        # 展平
        x = x.view(x.size(0), -1)  # [B, 512]
        
        # 分类
        out = self.classifier(x)   # [B, num_classes]
        
        return out

class ResBlock(nn.Module):
    ''' residual block '''
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        
        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.r1 = nn.ReLU(inplace=True)
        
        self.c2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.r2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H * W]
        

        out = self.c1(x)
        out = self.bn1(out)
        out = self.r1(out)
        out = self.c2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.r2(out)
        
        return out

class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()
        num_classes=10
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2. define multiple residual blocks
        self.layer1 = self._make_layer(in_channel=64, out_channel=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channel=64, out_channel=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channel=128, out_channel=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_channel=256, out_channel=512, num_blocks=2, stride=2)
        # 3. define full-connected layer to classify
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channel, out_channel, num_blocks, stride):
        '''
        辅助函数：构建包含多个 ResBlock 的层
        '''
        strides = [stride] + [1] * (num_blocks - 1) 
        # 例如 num_blocks=2, stride=2 -> strides=[2, 1]
        # 第一个 block 处理下采样 (stride=2)，后续 block 保持尺寸不变 (stride=1)
        layers = []
        for s in strides:
            layers.append(ResBlock(in_channel, out_channel, stride=s))
            in_channel = out_channel # 下一个 block 的输入通道等于上一个的输出
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * 3 * 32 * 32]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out) # [B, 64, 32, 32]
        out = self.layer2(out) # [B, 128, 16, 16]
        out = self.layer3(out) # [B, 256, 8, 8]
        out = self.layer4(out) # [B, 512, 4, 4]
        out = self.avg_pool(out) # [B, 512, 1, 1]
        out = torch.flatten(out, 1) # [B, 512]
        out = self.fc(out)       # [B, num_classes]
        
        return out


class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        bot_channels = int(round(out_channel/bottle_neck))
        self.c1 = nn.Conv2d(in_channel, bot_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bot_channels)
        self.r1 = nn.ReLU(inplace=True)
        
        self.c2 = nn.Conv2d(bot_channels, bot_channels, kernel_size=3, padding=1, stride=stride,groups=group, bias=False)
        self.bn2 = nn.BatchNorm2d(bot_channels)
        self.r2 = nn.ReLU(inplace=True)

        self.c3 = nn.Conv2d(bot_channels, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.r3 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.c1(x)
        out = self.bn1(out)
        out = self.r1(out)
        out = self.c2(out)
        out = self.bn2(out)
        out = self.r2(out)
        out = self.c3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.r3(out)
        return out


class ResNext(nn.Module):
    def __init__(self,):
        super().__init__()
        num_blocks=[3, 4, 6, 3]
        group=32
        bottle_neck_ratio=2
        num_classes=10
        self.group = group
        self.bottle_neck_ratio = bottle_neck_ratio
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Stage 1
        self.layer1 = self._make_layer(num_blocks[0], out_channel=256, stride=1)
        # Stage 2
        self.layer2 = self._make_layer(num_blocks[1], out_channel=512, stride=2)
        # Stage 3
        self.layer3 = self._make_layer(num_blocks[2], out_channel=1024, stride=2)
        # Stage 4
        self.layer4 = self._make_layer(num_blocks[3], out_channel=2048, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, num_blocks, out_channel, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            # 计算中间通道数
            layers.append(ResNextBlock(self.in_channels, out_channel, 
                                       bottle_neck=self.bottle_neck_ratio, group=self.group, stride=s))
            self.in_channels = out_channel
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out