import torch
import math

import copy

from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
          out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x,out], 1)
    

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)
    
class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers =[]
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class DenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
               reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        #in_planes = 16

        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 1st block
        n = 6
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        out_planes = int(math.floor(in_planes*reduction))
        self.trans1 = TransitionBlock(in_planes, out_planes, dropRate=dropRate)

        # 2st block
        n = 12
        self.block2 = DenseBlock(n, out_planes, growth_rate, block, dropRate)
        in_planes = int(out_planes+n*growth_rate)
        out_planes = int(math.floor(in_planes*reduction))
        self.trans2 = TransitionBlock(in_planes, out_planes, dropRate=dropRate)

        # 3rd block
        n = 24
        self.block3 = DenseBlock(n, out_planes, growth_rate, block, dropRate)
        in_planes = int(out_planes+n*growth_rate)
        out_planes = int(math.floor(in_planes*reduction))
        self.trans3 = TransitionBlock(in_planes, out_planes, dropRate=dropRate)

        # 4th block
        n = 16
        self.block4 = DenseBlock(n, out_planes, growth_rate, block, dropRate)
        in_planes = int(out_planes+n*growth_rate)

        self.final_bn = nn.BatchNorm2d(in_planes)
        self.relu_final = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(in_planes, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)        

    def forward(self, x, feature=False):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))      
        x = self.block4(x)
        x = self.relu_final(self.final_bn(x))

        x = F.adaptive_avg_pool2d(x, (1,1))
        
        x = torch.flatten(x, 1)
        if feature:
            return x
        
        x = self.fc(x)
        

        return x    
    
    
def densenet121(num_classes=1000, k=32):
    
    model = DenseNet(100, num_classes, k)
    
    return model