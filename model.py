import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split



class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(18432, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 5)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


#Replicado AlexNET
class TModel(nn.Module):

    def __init__(self):
        
        super(TModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 64, kernel_size= 5, stride=4, padding=1 )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 5, stride=2, padding=1 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride= 1, padding= 2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride= 1, padding= 2)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride= 1, padding= 1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride= 1, padding= 1)

        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.3)
        self.fc1  = nn.Linear(in_features= 6400, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 2048)
        self.fc3 = nn.Linear(in_features=2048 , out_features=5) #rango de edades

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.reshape(x.shape[0], -1)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class UTKFace(Dataset):

    def load_data(self, root_dir):
        entries = os.listdir(root_dir)
        X = []
        number_of_26 = 0
        yearin = -1
        rand = cont =  0
        for file in entries:
            attributes = file.split('_')
            year = int(attributes[0])
            img = cv2.imread(root_dir + file)
            if year >= 2 and year <= 18:
                yearin = 0
            elif year > 18 and year <= 30:
                if cont == 2:
                    cont = 0
                else:
                    yearin = 1
                    cont += 1
                
            elif year > 30 and year <= 45:
                yearin = 2
            elif year > 45 and year <= 60:
                yearin = 3
            elif year > 60 and year <= 90:
                yearin = 4
            
            if yearin != -1:
                X.append((img, yearin))
                yearin = -1

        n1 = n2 = n3 = n4 = n5 = n0 = 0
        for i in X:
            if i[1] == 0:
                n0 += 1
            if i[1] == 1:
                n1 += 1
            if i[1] == 2:
                n2 += 1
            if i[1] == 3:
                n3 += 1
            if i[1] == 4:
                n4 += 1

        
        print(n0)
        print(n1)
        print(n2)
        print(n3)
        print(n4)
        return X    
    
    def __init__(self, root_dir, transform=None):
        #read from datasource
        self.root_dir = root_dir
        self.samples = self.load_data(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        sample = self.samples[idx][0]

        #trans = transforms.ToTensor()

        #sample = trans(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.samples[idx][1]

    
def split_dataset(dataset, val_split=0.25):
    train_index, val_index = train_test_split(list(range(len(dataset))), test_size= val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_index)
    datasets['test'] = Subset(dataset, val_index)
    return datasets['train'], datasets['test']





"""
#model = AlexNETlike()
transform = transforms.ToTensor()
dataset = FaceLandmarksDataset(root_dir ='datasets/UTKFace/UTKFace/', transform = transform)

print(type(dataset[0]))
print(type(dataset[0][0]))

data, target = dataset[0]

print(data)
print(target)
print(len(dataset))

#print(model)
"""