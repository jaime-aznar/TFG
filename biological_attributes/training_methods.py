import torch
import copy
from torchvision import models
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #torch.cuda.empty_cache()
    print(device)
    model.to(device)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size


    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss


def train(epochs, model, train_split, test_split, loss_fn, optimizer):
    best_accuracy = 0.0
    best_loss = 200.0 
    steps_stopping = 0
    for epoch in range(epochs):
        print(f'=========== EPOCH: {epoch} ===========')
        print('TRAINING')
        train_loop(train_split, model, loss_fn, optimizer)
        print('TEST')
        acc, current_loss = test_loop(test_split, model, loss_fn)
        if best_loss > current_loss:
            best_loss = current_loss
            steps_stopping = 0
            best_accuracy = acc
            print(f'update best_model and change best_acc to: {acc}')
            best_model = copy.deepcopy(model)
        else:
            steps_stopping += 1
            print(f'EARLY STOPPED to prevent overfitting, best loss: {best_loss}')
            
        print(f'\n\n\nBest accuracy so far: {best_accuracy}\n\n\n')
        

    return best_model, best_accuracy


def get_model(out_features = 5, params = 90):
    model = models.efficientnet_b0(pretrained=True)
    cont = 0
    last = len(model.classifier) -  1

    num_features = model.classifier[last].in_features

    cont, cont1 = 0,0
    
    for param in model.parameters():
         
        if cont < params:
            cont += 1
            param.requires_grad = False
        else:
            cont1 += 1
    print(f'trainable params {cont1}\tnot trainable params {cont}')
    
    model.classifier[last] = nn.Linear(num_features, out_features)

    print(model)

    return model


def split_dataset(dataset, val_split=0.25):
    train_index, val_index = train_test_split(list(range(len(dataset))), test_size= val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_index)
    datasets['test'] = Subset(dataset, val_index)
    return datasets['train'], datasets['test']
