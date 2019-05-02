import torch as t 
from dataset import test_dataset,validation_dataset,train_dataset
from torch.utils.data import DataLoader
from torch import nn,optim
from torch.autograd import Variable
from resnet50 import Resnet50

batch_size=32
epoches = 20
learning_rate=0.1
num_class=20

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


model = Resnet50(num_class)
if t.cuda.is_available():
	mode = model.cuda()

criterion = nn.MultiLabelSoftMarginLoss()

optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-4,nesterov=True)

for epoch in range(epoches):
	running_loss = 0.0
	for data in train_loader:
		img,label = data
		img = Variable(img)
		label = Variable(label)

		if t.cuda.is_available():
			img = img.cuda()
			label = label.cuda()

		optimizer.zero_grad()

		out = model(img)
		loss = criterion(out,label)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

	print('epoch: {}, loss: {:.4f} '.format(epoch+1,running_loss))

	if epoch  in (5,10):
		model.eval()
		correct=0
		for data in validation_loader:
			img,label = data
			img = Variable(img)
			label = Variable(label)

			if t.cuda.is_available():
				img = img.cuda()
				label = label.cuda()

			out = model(img)
			predict = t.round(t.sigmoid(out))
			correct += (predict == label).sum()
		
		print("validation accuracy : %f " % (correct.item()/(len(validation_dataset)*num_class)))
		model.train()	

print("Finished Train!")

model.eval()
correct=0

for data in test_loader:
	img,label = data
	img = Variable(img)
	label = Variable(label)

	if t.cuda.is_available():
		img = img.cuda()
		label = label.cuda()

	out = model(img)
	predict = t.round(t.sigmoid(out))
	correct += (predict == label).sum()
		
print("test accuracy : %f" % (correct.item()/(len(test_dataset)*num_class)))
