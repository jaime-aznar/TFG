import multimodel
import vgg_occ_dataset as vgg_occlusion
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import challenge1.modelutk as m
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold

transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])
dataset = vgg_occlusion.VggFace2(samples = 240000, path = '../datasets/VGG-Face2/', csvfile= '../datasets/MAAD_Face.csv', transform = transform)

splits = KFold(n_splits = 10, shuffle = True,random_state=77)

#train test splits
"""
train_split, test_split = m.split_dataset(dataset, val_split=0.1)
train_dataloader = DataLoader(train_split, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=16, shuffle=False)
"""

"""
cont = 0
for param in model.parameters():
    if cont < 60 and param.requires_grad == True:
        param.requires_grad = False
        cont += 1

epochs = 25
"""


#TODO: adjust class weights 

criterion1 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.eyeglasses]).to('cuda'))
criterion2 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.beard]).to('cuda'))
criterion3 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.hat]).to('cuda'))


criterion = [criterion1, criterion2, criterion3]

best_model = None

best_loss = np.Inf

losses = []

"""
for i in range(10):
    model = multimodel.MultiCNN(True)
    model.setTrainableParams(60)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_trained, loss = multimodel.train(epochs, model, train_dataloader, test_dataloader, criterion, optimizer)
    losses.append(loss)

    if loss < best_loss:
        best_loss = loss
        best_model = model_trained
"""
epochs = 12
for train_idx, test_idx in splits.split(np.arange(len(dataset))):

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)

    model = multimodel.MultiCNN(True)
    model.setTrainableParams(60)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_trained, loss = multimodel.train(epochs, model, train_loader, test_loader, criterion, optimizer)
    losses.append(loss)

    if loss < best_loss:
        best_loss = loss
        best_model = model_trained


torch.save(best_model.state_dict(), 'best_3occ_model.pth')

f = open('write_loss_multimodel.txt', 'w')

for i in losses:
    f.write(str(i) + '\n')

f.close()

