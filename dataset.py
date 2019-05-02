import os
from PIL import Image
import numpy as np 
import torch as t
from torchvision import transforms 
from torch.utils import data
#import matplotlib.pyplot as plt

class_path = '/home/zhoutao/myvoc2012/tag_dict.txt'

class_dict = {}
class_num=0
with open(class_path) as f:
	lines = f.readlines()
	for line in lines:
		class_dict[line.strip()]=class_num
		class_num+=1


transform = transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
])

class Vocdata(data.Dataset):
	def __init__(self,root,transforms=None):
		self.root = root
		self.inf = self._read_inf(root+'/tags.txt')
		self.transforms = transforms

	def _read_inf(self,path):
		with open(path) as p:
			return p.readlines()

	def __getitem__(self,index):
		img,tag = self.inf[index].split('*')
		tags = tag.strip().split(' ')
		label = t.zeros(class_num)

		for  i in tags:
			label[class_dict[i]]=1.0

		data = Image.open(os.path.join(self.root,'images',img))
		if self.transforms:
			data = self.transforms(data)

		return data,label

	def __len__(self):
		return len(self.inf)

test_dataset = Vocdata('/home/zhoutao/myvoc2012/test',transforms=transform)
validation_dataset = Vocdata('/home/zhoutao/myvoc2012/validation',transforms=transform)
train_dataset = Vocdata('/home/zhoutao/myvoc2012/train',transforms=transform)

'''
---------------test part--------------------------------------

test_data = Vocdata('D:/myvoc2012/test',transforms=transform)
img1, label1 = test_data[4]

print(img1.size(),label1)
img2=transforms.ToPILImage()(img1)
plt.imshow(img2)
plt.show()

'''
