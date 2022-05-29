from tkinter import N

from sklearn.utils import shuffle
import vgg_dataset as v
import utk_dataset as m
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn as nn
import copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = 0



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
    loss_train, loss_test = [], []
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
            break
            if steps_stopping == 1: #if the algorithm doesnt improve loss two epochs in a row => STOP to prevent overfitting

                break
        print(f'\n\n\nBest accuracy so far: {best_accuracy}\n\n\n')
        

    return best_model, best_accuracy


def get_model(out_features = 5, params = 90):
    #model = models.efficientnet_b0(pretrained=True)
    cont = 0
    model = models.mobilenet_v3_large(pretrained= True)
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
    


    #model.classifier[last] = nn.Sequential(nn.Dropout(), nn.Linear(num_features, out_features))

    model.classifier[last] = nn.Linear(num_features, out_features)

    print(model)

    return model



"""
if data == 2:
    print('updating class weights due to dataset imbalance')
    n0,n1,n2,n3 = 10078, 4526, 3434 ,3975#, 1692
    weights = [n0/n0, n0/n1, n0/n2, n0/n3]#, n0/n4]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
else: 
    loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor([16434/3754,16434/16434, 16434/2310]).to(device))
    #loss_fn = nn.CrossEntropyLoss()
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
])



#dataset = m.UTKFace(root_dir ='../datasets/UTKFace/', transform = transform, extract=data)
dataset = v.VggFace2(path='../datasets/VGG-Face2/', csvfile= '../datasets/MAAD_Face.csv', transform = transform, extract = 'Race')
print(f'0 {dataset.n0} 1 {dataset.n1} 2 {dataset.n2}')
train_split, test_split = m.split_dataset(dataset, val_split=0.1)
train_dataloader = DataLoader(train_split, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_split, batch_size=32, shuffle=False)


loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor([dataset.n2/dataset.n0, dataset.n2/dataset.n1, dataset.n2/dataset.n2]).to(device))




epochs = 10

learning_rate = 0.001



best_acc = 0.0
best_model = None
iterations = 0
best_iter = -1

accuracy_1, accuracy_2 = [], []
"""
while best_acc < 0.84 and iterations < 8:
    
    model = get_model()
    optimizer = None
    if iterations >= 4:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    model_trained, accuracy = train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)
    
    accuracy_1.append(accuracy)

    if accuracy > best_acc:
        best_iter = iterations
        best_acc = accuracy
        best_model = copy.deepcopy(model_trained)

    iterations += 1

print(f'best accuracy stored is: {best_acc}')
if best_iter < 4:
    print('optimizer: sgd 0.001')
else:
    print('optimizer: sgd 0.01')

"""

iterations = 0
best_iter = -1
best_model2 = None
best_acc2 = 0.00
print('Beginning training')

while best_acc2 < 0.99 and iterations < 5:
    model = get_model(out_features=3, params = 24)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    
    model_trained, accuracy = train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)
    
    accuracy_2.append(accuracy)

    if accuracy > best_acc2:
        best_iter = iterations
        best_acc2 = accuracy
        best_model2 = copy.deepcopy(model_trained)

    iterations += 1

print(f'best accuracy stored is: {best_acc2}')


torch.save(best_model2.state_dict(), 'model_mobilenet_race_vgg.pth')


f = open('write_accuracies_race_mobile_vgg.txt','w')

for i in accuracy_2:
    f.write(str(i) + '\n')

f.close()
