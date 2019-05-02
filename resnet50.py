#@Time 2019-5-1
#@Author zzz

import torch as t 
import torch.nn as nn



class Bottleneck(nn.Module):
	def __init__(self,inchannel,midchannel,outchannel,stride=1,downsample=None):
		super(Bottleneck,self).__init__()
		self.conv1 = nn.Conv2d(inchannel,midchannel,kernel_size=1,stride=1,bias=False)
		self.bn12 = nn.BatchNorm2d(midchannel)
		self.conv2 = nn.Conv2d(midchannel,midchannel,kernel_size=3,stride=stride,padding=1,bias=False)
		self.conv3 = nn.Conv2d(midchannel,outchannel,kernel_size=1,stride=1,bias=False)
		self.bn3 = nn.BatchNorm2d(outchannel)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride
		self.downsample = downsample

	def forward(self,x):
		identity=x

		out = self.conv1(x)
		out = self.bn12(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn12(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(identity)
			identity = self.bn3(identity)

		out += identity
		out = self.relu(out)

		return out


class Resnet50(nn.Module):
	def __init__(self,num_classes):
		super(Resnet50,self).__init__()
		self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.layer1 = self._make_layer(64,64,256,basic_blocks=2)
		self.layer2 = self._make_layer(256,128,512,basic_blocks=3,stride=2)
		self.layer3 = self._make_layer(512,256,1024,basic_blocks=5,stride=2)
		self.layer4 = self._make_layer(1024,512,2048,basic_blocks=2,stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(2048,num_classes)

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
			elif isinstance(m,nn.BatchNorm2d):
				nn.init.constant_(m.weight,1)
				nn.init.constant_(m.bias,0)

	def _make_layer(self,inchannel,midchannel,outchannel,basic_blocks,stride=1):
		layers = []

		downsample = nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False)
		layers.append(Bottleneck(inchannel,midchannel,outchannel,stride,downsample))
		for _ in range(basic_blocks):
			layers.append(Bottleneck(outchannel,midchannel,outchannel))

		return nn.Sequential(*layers)

	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0),-1)
		x = self.fc(x)

		return x
