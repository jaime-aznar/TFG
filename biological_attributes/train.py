from tkinter import N
from sklearn.utils import shuffle
import vgg_dataset as v
import utk_dataset as m
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
import copy
import training_methods as trm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#0 for age, 1 for gender, 2 for race
data = 0

#True --> Load Utk
utk = False

#specify data location
data_directory = '../../Work/UTKFace/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
])


if utk:
    dataset = m.UTKFace(root_dir = data_directory, transform = transform, extract=data)

    if data == 2:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss() 

else:
    dataset = v.VggFace2(path='../datasets/VGG-Face2/', csvfile= '../datasets/MAAD_Face.csv', transform = transform, extract = 'Race')
    print(f'0 {dataset.n0} 1 {dataset.n1} 2 {dataset.n2}')
    #data class adjustment
    loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor([dataset.n2/dataset.n0, dataset.n2/dataset.n1, dataset.n2/dataset.n2]).to(device))

epochs = 10

iterations = 0
best_iter = -1

accuracy_2 = []

iterations = 0
best_iter = -1
best_model2 = None
best_acc2 = 0.00
print('Beginning training')

while best_acc2 < 0.99 and iterations < 10:
    #get model
    model = trm.get_model(out_features=3, params = 24)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #get different split
    train_split, test_split = trm.split_dataset(dataset, val_split=0.1)
    train_dataloader = DataLoader(train_split, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_split, batch_size=32, shuffle=False)

    #train current model
    model_trained, accuracy = trm.train(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer)
    
    accuracy_2.append(accuracy)

    if accuracy > best_acc2:
        best_iter = iterations
        best_acc2 = accuracy
        best_model2 = copy.deepcopy(model_trained)

    iterations += 1

print(f'best accuracy stored is: {best_acc2}')


torch.save(best_model2.state_dict(), 'model_race_eff.pth')


f = open('accs_race_eff.txt','w')

for i in accuracy_2:
    f.write(str(i) + '\n')

f.close()
