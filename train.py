from tkinter import N

from sklearn.utils import shuffle
import model as m
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import confusion_matrix
#from resources.plotcm import plot_confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

data = 0

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, test_accuracy = None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' - Model Test Accuracy = ' + str(round(test_accuracy, 3)))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([])
    all_labels = all_labels.to(device)
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
        all_labels = torch.cat((all_labels, labels),
            dim = 0)

    return all_preds, all_labels

def print_confussion_matrix(model, data, acc):
    conf = DataLoader(data, batch_size=128)
    preds, labels = get_all_preds(model, conf)
    preds = preds.to('cpu')
    labels = labels.to('cpu')
    cm = confusion_matrix(labels, preds.argmax(dim=1))
    
    #plot_confusion_matrix(cm, classes=['Hispanic','Black','Asian','Indian','Other'], test_accuracy = acc)
    plot_confusion_matrix(cm, classes=['[2-18]', '[18-30]', '[30-45]', '[45-60]', '[60-90]'], test_accuracy = acc)


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



def test_loop(dataloader, model, loss_fn, best_accuracy):
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

    if best_accuracy < correct:
        best_accuracy = correct

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return best_accuracy, test_loss


def train(epochs, model, train_split, test_split, loss_fn, optimizer):
    best_accuracy = 0.0
    best_loss = 40.0 
    steps_stopping = 0
    for epoch in range(epochs):
        print(f'=========== EPOCH: {epoch} ===========')
        print('TRAINING')
        train_loop(train_split, model, loss_fn, optimizer)
        print('TEST')
        acc, current_loss = test_loop(test_split, model, loss_fn, best_accuracy)
        if best_loss > current_loss:
            best_loss = current_loss
            steps_stopping = 0
            if acc != best_accuracy:
                best_accuracy = acc
                print(f'change acc: {acc}')
                best_model = copy.deepcopy(model)
        
        else:
            steps_stopping += 1
            if steps_stopping == 2: #if the algorithm doesnt improve loss two epochs in a row => STOP to prevent overfitting
                print(f'EARLY STOPPED to prevent overfitting, best loss: {best_loss}')
                break
        print(f'\n\n\nBest accuracy so far: {best_accuracy}\n\n\n')
        

    return best_model, best_accuracy




def set_parameter_requires_grad(model, feature_extracting = True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def show_parameter_requires_grad(model, feature_extracting = True):
    if feature_extracting:
        for param in model.parameters():
            if param.requires_grad == True:
                print('yes')





model = models.efficientnet_b0(pretrained=True)

#set_parameter_requires_grad(model)


last = len(model.classifier) -  1

num_features = model.classifier[last].in_features

cont = 0

for param in model.parameters():
    cont += 1 
    if cont < 180:
        param.requires_grad = False

print(cont)
print(num_features)

model.classifier[last] = nn.Linear(num_features,5)

print(model)

epochs = 10

learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

if data == 2:
    print('updating class weights due to dataset imbalance')
    n0,n1,n2,n3,n4 = 10078, 4526, 3434,3975, 1692
    weights = [n0/n0, n0/n1, n0/n2, n0/n3, n0/n4]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
else: 
    loss_fn = nn.CrossEntropyLoss()


dataset = m.UTKFace(root_dir ='../UTKFace/', transform = transform, extract=data)
train_split, test_split = m.split_dataset(dataset, val_split=0.1)
train_dataloader = DataLoader(train_split, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=32, shuffle=False)


cont = 0

for param in model.parameters():
    if param.requires_grad:
        cont += 1

print(f'trainable: {cont}')

best_model, accuracy = train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)


print_confussion_matrix(best_model, dataset, accuracy)