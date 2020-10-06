# %%time
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:01:13 2020

@author: Yun, Junhyuk
"""
# https://arxiv.org/abs/1202.2745
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torchvision import datasets, transforms
from torch.autograd import Variable

# GPU settings
device = 'cuda' if cuda.is_available() else 'cpu'
print("Device: ", device, "\n");

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)        
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(360, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        in_size = x.size(0)
        x=self.conv1(x)   # L1 26*26*20
        x=self.mp1(x)     # L2 13*13*20
        x=self.conv2(x)   # L3 9*9*40
        x=self.mp2(x)     # L4 3*3*40
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc1(x)   # L5 150
        x=self.fc2(x)     # L6 10
        return F.log_softmax(x)


model = Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = F.nll_loss
# criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set({}): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test(epoch)
