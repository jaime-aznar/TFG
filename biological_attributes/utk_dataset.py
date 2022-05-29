import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2




#Replicado AlexNET - NO USADO
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
        self.fc3 = nn.Linear(in_features=2048 , out_features=2) #rango de edades

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
    
    def load_data(self, root_dir, extract):
        entries = os.listdir(root_dir)
        X = []
        yearin = -1
        cont =  0

        if extract == 0: #read age data
            n0,n1,n2 = 0,0,0
            for file in entries:
                mode = False
                
                attributes = file.split('_')
                year = int(attributes[0])
                img = cv2.imread(root_dir + file)
                if mode:
                    if year >= 2 and year <= 90:
                        yearin = year
                else:
                    if year >= 2 and year <= 20:
                        yearin = 0
                        n0 += 1
                    elif year > 20 and year <= 60:
                        yearin = 1
                        n1 += 1
                    elif year > 60 and year <= 90:
                        yearin = 2
                        n2 += 1

                if yearin != -1:
                    X.append((img, yearin))
                    yearin = -1

            print(f'0 {n0}\n1 {n1}\n2 {n2}')

        elif extract == 1: #read gender data
            n0, n1 = 0, 0
            for file in entries:
                attributes = file.split('_')
                age = int(attributes[1])
                img = cv2.imread(root_dir + file)
                if age == 0 and n0 > 11317:
                    n0 = n0 #dont do anything
                else:
                    X.append((img, age))

        elif extract == 2: #read race data
            n0,n1,n2,n3,n4 = 0,0,0,0,0
            for file in entries:
                
                attributes = file.split('_')
                img = cv2.imread(root_dir + file)

                try:
                    race = int(attributes[2])
                    if race == 0:
                        n0 += 1
                    elif race == 1:
                        n1 += 1
                    elif race == 2:
                        n2 += 1
                    elif race == 3:
                        n3 += 1
                    elif race == 4:
                        n4 += 1
                    if race == 4: #skip Others class
                        continue
                    else:
                        X.append((img, race))
                
                except Exception:
                    print(f'Exception raised on filename: {file}')
                
                    if cont == 0 or cont == 2: #black race subjects
                        race = 1
                    
                    X.append((img, race))

                    cont += 1

            print(f'{n0}\n{n1}\n{n2}\n{n3}\n{n4}\n')

        return X    
    
    def __init__(self, root_dir, transform=None, extract = 0):
        #read from datasource
        self.root_dir = root_dir
        self.samples = self.load_data(root_dir, extract)
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

    




