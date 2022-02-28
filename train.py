import model as m
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        print("ey")
        pred = model(X)
        print("ey")
        loss = loss_fn(pred, y)
        print("ey")
        # Backpropagation
        optimizer.zero_grad()
        print("ey")
        loss.backward()
        print("ey")
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    cont = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(epochs, model, train_split, test_split, loss_fn, optimizer):
    for epoch in range(epochs):
        train_loop(train_split, model, loss_fn, optimizer)
        print('Test with test split: ')
        test_loop(test_split, model, loss_fn)
        print('Test with train split: ')
        test_loop(train_split, model, loss_fn)





model = models.vgg16(pretrained=True)


last = len(model.classifier) - 1

num_features = model.classifier[last].in_features


model.classifier[last] = nn.Linear(num_features, 6)




print(model)

epochs = 20

learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

optimizer = torch.optim.Adam(model.classifier[last].parameters(), lr = learning_rate)
loss_fn = nn.CrossEntropyLoss()

dataset = m.UTKFace(root_dir ='datasets/UTKFace/UTKFace/', transform = transform)
train_split, test_split = m.split_dataset(dataset, val_split=0.1)
train_dataloader = DataLoader(train_split, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=64, shuffle=False)

train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)


"""
model = m.VGG16()

epochs = 15

learning_rate = 0.005

transform = transforms.ToTensor()

transformAug = transforms.Compose
([      
        transforms.RandomHorizontalFlip(p = 0.3), 
        #transforms.RandomAdjustSharpness(sharpness_factor = 1.5, p = 0.2),
        #transforms.RandomSolarize(threshold=192.0),
        transforms.ToTensor()
])


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
loss_fn = nn.CrossEntropyLoss()



dataset = m.UTKFace(root_dir ='datasets/UTKFace/UTKFace/', transform = transform)
train_split, test_split = m.split_dataset(dataset, val_split=0.1)
train_dataloader = DataLoader(train_split, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=64, shuffle=False)


print('Dataloaders done!')






#model2 = m.AlexNETlike()
#optimizer = torch.optim.SGD(model2.parameters(), lr = learning_rate)

train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)
"""