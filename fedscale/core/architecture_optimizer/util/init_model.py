import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torch import Tensor
from typing import Optional
from torchvision.models.resnet import conv1x1, conv3x3


class OptBlock(nn.Module):
    """
    architecture:
    (-> conv -> bn -> relu ->) * n
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.convs = nn.ModuleList()
        self.convs.append(conv3x3(inplanes, planes, stride=stride))
        self.convs.append(conv3x3(planes, planes))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm2d(planes))
        self.bns.append(nn.BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def getOutChannel(self, idx: int) -> int:
        return self.convs[idx].out_channels

    def getInChannel(self, idx: int) -> int:
        return self.convs[idx].out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = x

        for i, conv in enumerate(self.convs):
            # -> conv -> bn -> relu ->
            out = conv(out)
            # print(f"out={out.shape}")
            out = self.bns[i](out)
            if i != self.convs.__len__() - 1:
                out = self.relu(out)

        if self.downsample != None:
            identity = self.downsample(identity)

        try:
            out = out + identity
        except Exception as error:
            print(f"out={out.shape}, identity={identity.shape}")
            print(error)
            print(self.convs)
            print(self.downsample)
            exit(1)

        out = self.relu(out)

        return out

    def widenConvOut(self, idx: int, newOutChannelSize: int, randomMap: list) -> None:
        oldConv = self.convs[idx]
        if newOutChannelSize <= oldConv.out_channels:
            raise ValueError('Not support narrow down a layer.')
        if newOutChannelSize != len(randomMap):
            raise ValueError('Invalid mapping.')
        oldBn = self.bns[idx]
        # self.convs[idx] = conv3x3(
        #     oldConv.in_channels, newOutChannelSize, stride=oldConv.stride)
        self.convs[idx] = nn.Conv2d(oldConv.in_channels, newOutChannelSize, 
            kernel_size=oldConv.kernel_size[0], stride=oldConv.stride, bias=False, padding=oldConv.padding, groups=1, dilation=1)
        self.bns[idx] = nn.BatchNorm2d(newOutChannelSize)
        convParam = OrderedDict()
        bnParam = OrderedDict()
        convParam['weight'] = torch.zeros(
            newOutChannelSize, oldConv.in_channels, oldConv.kernel_size[0], oldConv.kernel_size[1])
        bnParam['weight'] = torch.zeros(newOutChannelSize)
        bnParam['bias'] = torch.zeros(newOutChannelSize)
        bnParam['running_mean'] = torch.zeros(newOutChannelSize)
        bnParam['running_var'] = torch.zeros(newOutChannelSize)
        for i in range(newOutChannelSize):
            convParam['weight'][i, :, :, :] = oldConv.state_dict()[
                'weight'][randomMap[i], :, :, :]
            bnParam['weight'][i] = oldBn.state_dict()['weight'][randomMap[i]]
            bnParam['bias'][i] = oldBn.state_dict()['bias'][randomMap[i]]
            bnParam['running_mean'][i] = oldBn.state_dict()[
                'running_mean'][randomMap[i]]
            bnParam['running_var'][i] = oldBn.state_dict()[
                'running_var'][randomMap[i]]
        self.convs[idx].load_state_dict(convParam)
        self.bns[idx].load_state_dict(bnParam)

        if idx == len(self.convs) - 1:
            self.widenDownsampleOut(newOutChannelSize, randomMap)

    def widenConvIn(self, idx: int, newInChannelSize: int, randomMap: list) -> None:
        oldConv = self.convs[idx]
        if newInChannelSize < oldConv.in_channels:
            raise ValueError('Not support narrow down a layer.')
        if newInChannelSize != len(randomMap):
            raise ValueError('Invalid Mapping')

        # functionality preserving
        scale = []
        for i in randomMap:
            scale.append(randomMap.count(i))

        # self.convs[idx] = conv3x3(
        #     newInChannelSize, oldConv.out_channels, stride=oldConv.stride)
        self.convs[idx] = nn.Conv2d(newInChannelSize, oldConv.out_channels,
            kernel_size=oldConv.kernel_size[0], stride=oldConv.stride, bias=False, padding=oldConv.padding, groups=1, dilation=1)
        convParam = OrderedDict()
        convParam['weight'] = torch.zeros(
            oldConv.out_channels, newInChannelSize, oldConv.kernel_size[0], oldConv.kernel_size[1])
        for i in range(newInChannelSize):
            convParam['weight'][:, i, :, :] = oldConv.state_dict()['weight'][:, randomMap[i], :, :] / scale[i]
        self.convs[idx].load_state_dict(convParam)

        if idx == 0:
            self.widenDownsampleIn(newInChannelSize, randomMap)

    def widenDownsampleOut(self, newOutChannelSize: int, randomMap: list) -> None:
        if self.downsample == None:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, newOutChannelSize, 1),
                nn.BatchNorm2d(newOutChannelSize)
            )
        oldDownsample = self.downsample
        inChannelSize = oldDownsample[0].in_channels
        stride = oldDownsample[0].stride
        self.downsample = nn.Sequential(
            conv1x1(inChannelSize, newOutChannelSize, stride),
            nn.BatchNorm2d(newOutChannelSize)
        )
        convParam = OrderedDict()
        bnParam = OrderedDict()
        convParam['weight'] = torch.zeros(
            newOutChannelSize, inChannelSize, 1, 1)
        bnParam['weight'] = torch.zeros(newOutChannelSize)
        bnParam['bias'] = torch.zeros(newOutChannelSize)
        bnParam['running_mean'] = torch.zeros(newOutChannelSize)
        bnParam['running_var'] = torch.zeros(newOutChannelSize)
        for i in range(newOutChannelSize):
            convParam['weight'][i, :, :, :] = oldDownsample[0].state_dict()[
                'weight'][randomMap[i], :, :, :]
            bnParam['weight'][i] = oldDownsample[1].state_dict()[
                'weight'][randomMap[i]]
            bnParam['bias'][i] = oldDownsample[1].state_dict()[
                'bias'][randomMap[i]]
            bnParam['running_mean'][i] = oldDownsample[1].state_dict()[
                'running_mean'][randomMap[i]]
            bnParam['running_var'][i] = oldDownsample[1].state_dict()[
                'running_var'][randomMap[i]]
        self.downsample[0].load_state_dict(convParam)
        self.downsample[1].load_state_dict(bnParam)

    def widenDownsampleIn(self, newInChannelSize: int, randomMap: list):
        if self.downsample == None:
            self.downsample = nn.Sequential(
                conv1x1(newInChannelSize, self.planes, 1),
                nn.BatchNorm2d(self.planes)
            )
        self.inplanes = newInChannelSize
        oldDownsample = self.downsample
        outChannelSize = oldDownsample[0].out_channels
        stride = oldDownsample[0].stride
        self.downsample = nn.Sequential(
            conv1x1(newInChannelSize, outChannelSize, stride),
            nn.BatchNorm2d(outChannelSize)
        )
        self.downsample[1].load_state_dict(oldDownsample[1].state_dict())

        # functionality preserving
        scale = []
        for i in randomMap:
            scale.append(randomMap.count(i))

        convParam = OrderedDict()
        convParam['weight'] = torch.zeros(
            oldDownsample[0].out_channels, newInChannelSize, 1, 1)
        for i in range(newInChannelSize):
            convParam['weight'][:, i, :, :] = oldDownsample[0].state_dict(
            )['weight'][:, randomMap[i], :, :] / scale[i]
        self.downsample[0].load_state_dict(convParam)

    def insertLayer(self, idx: int, kernelSize: int, stride: int) -> None:
        padding = 0
        if kernelSize == 3:
            padding = 1
        outChannelsize = self.convs[idx].out_channels
        if idx == self.convs.__len__() - 1:
            self.convs.append(nn.Conv2d(outChannelsize, outChannelsize,
                          kernelSize, stride, bias=False, padding=padding, groups=1, dilation=1))
            self.bns.append(nn.BatchNorm2d(outChannelsize))
        else:
            self.convs.insert(idx+1, nn.Conv2d(outChannelsize, outChannelsize,
                          kernelSize, stride, bias=False, padding=padding, groups=1, dilation=1))
            self.bns.insert(idx+1, nn.BatchNorm2d(outChannelsize))
        nn.init.dirac_(self.convs[idx+1].weight)
        nn.init.constant_(self.bns[idx+1].weight, 1)
        nn.init.constant_(self.bns[idx+1].bias, 0)


class OptResnet18 (nn.Module):
    """
    architecture:
    conv1 -> bn1 -> relu -> maxpool ->
    layer1 -> layer2 -> layer3 -> layer4 ->
    averagepool -> fc

    structure of layer(i):
    block1 -> block2

    struct of block(i):
    conv1 -> bn1 -> relu -> conv2 -> bn2 -> +id -> relu
    """

    def __init__(self, numClasses: int = 10, seed=233, widenNum=16) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [self._makeLayer(64, 2),
             self._makeLayer(128, 2),
             self._makeLayer(256, 2),
             self._makeLayer(512, 2)]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, numClasses)

        self.layersdict = [
            [2, 2],
            [2, 2],
            [2, 2],
            [2, 2]
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        random.seed(seed)
        self.widenNum = widenNum

    def _makeLayer(self, planes: int, stride: int = 1) -> nn.ModuleList:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        layers = nn.ModuleList()
        layers.append(OptBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        layers.append(OptBlock(self.inplanes, planes))
        return layers

    def _widenFc(self, newInFeatureSize: int,  randomMap: list) -> None:
        oldFc = self.fc
        self.fc = nn.Linear(newInFeatureSize, oldFc.out_features)
        fcParam = OrderedDict()
        fcParam['weight'] = torch.zeros(
            self.fc.out_features, self.fc.in_features)
        fcParam['bias'] = oldFc.state_dict()['bias']
        for i in range(newInFeatureSize):
            fcParam['weight'][:, i] = oldFc.state_dict()['weight'][:,
                                                                   randomMap[i]]
        self.fc.load_state_dict(fcParam)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in self.layers:
            for layer in block:
                x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def widden(self, idx: int) -> None:
        layer_idx = -1
        block_idx = -1
        for l in range(4):
            if sum(self.layersdict[l]) > idx:
                layer_idx = l
                if self.layersdict[l][0] > idx:
                    block_idx = 0
                    break
                else:
                    block_idx = 1
                    idx -= self.layersdict[l][0]
                    break
            else:
                idx -= sum(self.layersdict[l])
        if layer_idx == -1 or block_idx == -1:
            raise ValueError('Invalid layer to widen.')
        randomMap = list(
            range(self.layers[layer_idx][block_idx].getOutChannel(idx)))
        extraMap = random.choices(randomMap, k=self.widenNum)
        randomMap = randomMap + extraMap
        widenSize = len(randomMap)
        self.layers[layer_idx][block_idx].widenConvOut(
            idx, widenSize, randomMap)
        if idx < len(self.layers[layer_idx][block_idx].convs)-1:
            self.layers[layer_idx][block_idx].widenConvIn(
                idx+1, widenSize, randomMap)
        elif block_idx < 1:
            self.layers[layer_idx][block_idx +
                                   1].widenConvIn(0, widenSize, randomMap)
        elif layer_idx < 3:
            self.layers[layer_idx+1][0].widenConvIn(0, widenSize, randomMap)
        else:
            self._widenFc(widenSize, randomMap)

    def deepen(self, idx: int, kernelSize: int, stride: int) -> None:
        layer_idx = -1
        block_idx = -1
        for l in range(4):
            if sum(self.layersdict[l]) > idx:
                layer_idx = l
                if self.layersdict[l][0] > idx:
                    block_idx = 0
                    break
                else:
                    block_idx = 1
                    idx -= self.layersdict[l][0]
                    break
            else:
                idx -= sum(self.layersdict[l])
        if layer_idx == -1 or block_idx == -1:
            raise ValueError('Invalid layer to deepen.')
        if layer_idx == 0 and kernelSize > 4:
            kernelSize = 3
        elif layer_idx == 1 and kernelSize > 4:
            kernelSize = 3
        elif layer_idx == 2 and kernelSize > 2:
            kernelSize = 1
        elif layer_idx == 3 and kernelSize > 1:
            kernelSize = 1
        self.layers[layer_idx][block_idx].insertLayer(idx, kernelSize, 1)
        self.layersdict[layer_idx][block_idx] += 1


def init_model(name: str) -> nn.Module:
    if name == 'resnet18':
        return OptResnet18(numClasses=10)
    else:
        raise NotImplementedError
