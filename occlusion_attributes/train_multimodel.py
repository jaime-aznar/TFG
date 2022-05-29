import multimodel
import vgg_occ_dataset as vgg_occlusion
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold

#specify data directories
data_directory = '../datasets/VGG-Face2/'
csv_directory = '../datasets/MAAD_Face.csv'

transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])
dataset = vgg_occlusion.VggFace2(samples = 240000, path = data_directory , csvfile= csv_directory, transform = transform)

splits = KFold(n_splits = 10, shuffle = True,random_state=77)


criterion1 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.eyeglasses]).to('cuda'))
criterion2 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.beard]).to('cuda'))
criterion3 = nn.CrossEntropyLoss(weight = torch.FloatTensor([1.0, len(dataset)/dataset.hat]).to('cuda'))


criterion = [criterion1, criterion2, criterion3]

best_model = None

best_loss = np.Inf

losses = []

epochs = 12
for train_idx, test_idx in splits.split(np.arange(len(dataset))):

    #get fold
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=16, sampler=test_sampler)
    #define model
    model = multimodel.MultiCNN(True)
    model.setTrainableParams(60)

    #get optimizer with model params
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #train model

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

