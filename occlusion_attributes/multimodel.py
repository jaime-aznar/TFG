import pretrainedmodels
import torch.nn as nn
import torch
from torch.nn import functional as F
import copy
import numpy as np

device = 'cuda'

class MultiCNN(nn.Module):

    def __init__(self, pretrained):
        super(MultiCNN, self).__init__()

        if pretrained == True:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)


        self.fc1 = nn.Linear(512,2)
        self.fc2 = nn.Linear(512,2)
        self.fc3 = nn.Linear(512,2)

    def setTrainableParams(self, params):
        cont = 0
        for param in self.model.parameters():            
            param.requires_grad = False
            cont += 1
            if cont == params:
                break   

    #override forward function to return three predictions
    def forward(self, x):
        bs, _, _, _ = x.shape
        #execute model until output layers
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        label_eyeglasses = self.fc1(x)
        label_beard = self.fc2(x)
        label_hat = self.fc3(x)
        #return 3 predictions
        return{'label_eyeglasses': label_eyeglasses, 'label_beard': label_beard, 'label_hat': label_hat} 



def train_loop(epoch, dataloader, model, loss_fn, optimizer):
    print(device)
    model.to(device)
    model.train()
    train_loss = 0.0
    length = len(dataloader)
    print(device)
    for batch, (X, eyes, beard, hat) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        eyes = eyes.to(device)
        beard = beard.to(device)
        hat = hat.to(device)
        pred = model(X)
        pred_eyes = pred['label_eyeglasses']
        pred_beard = pred['label_beard']
        pred_hat = pred['label_hat']

        loss1 = loss_fn[0](pred_eyes, eyes) #eyes.squeeze().type(torch.LongTensor).to(device))
        loss2 = loss_fn[1](pred_beard, beard)
        loss3 = loss_fn[2](pred_hat, hat)

        loss = loss1 + loss2 + loss3
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = train_loss + ((1 / (batch + 1)) * (loss.data - train_loss))

        if batch % 100 == 0:
            print(f"loss: {train_loss:>7f}\t [{epoch}-{batch}/{length}]")



def test_loop(dataloader, model, loss_fn):
    model.to(device)
    valid_loss = 0.0
    with torch.no_grad():
        model.eval()
        for batch, (X, eyes, beard, hat) in enumerate(dataloader):
            X = X.to(device)
            eyes = eyes.to(device)
            beard = beard.to(device)
            hat = hat.to(device)

            pred = model(X)
            pred_eyes = pred['label_eyeglasses']
            pred_beard = pred['label_beard']
            pred_hat = pred['label_hat']

            loss1 = loss_fn[0](pred_eyes, eyes)
            loss2 = loss_fn[0](pred_beard, beard)
            loss3 = loss_fn[0](pred_hat, hat)

            loss = loss1 + loss2 + loss3
            valid_loss = valid_loss + ((1 / (batch + 1)) * (loss.data - valid_loss))



    print(f'Validation loss: {valid_loss:>6f}')
    return valid_loss




def train(epochs, model, train_dataloader, test_dataloader, criterion, optimizer):
    best_accuracy = 0.0
    best_loss = np.Inf 

    for epoch in range(epochs):
        print(f'=========== EPOCH: {epoch} ===========')
        
        print('TRAINING')
        train_loop(epoch, train_dataloader, model, criterion, optimizer)
        
        print('VALIDATION')
        current_loss = test_loop(test_dataloader, model, criterion)

        if best_loss > current_loss:
            best_loss = current_loss
            print(f'update best_model and best accuracy ({current_loss:>6f})')
            best_model = copy.deepcopy(model)
        else:
            print(f'EARLY STOPPED to prevent overfitting, best loss: {best_loss}')
            break

        print(f'\n\n\nBest loss so far: {best_loss}\n\n\n')
        

    return best_model, best_loss

