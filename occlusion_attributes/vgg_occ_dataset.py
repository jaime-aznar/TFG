import csv
from logging.config import fileConfig
from torch.utils.data import Dataset
import os
from fnmatch import fnmatch
import cv2
from random import randint
from random import seed
import torch

class VggFace2(Dataset):
    
    def __init__(self, path, csvfile, samples = 650000, transform = None):
        print('Loading Dataset')
        self.samples = []
        self.beard = 0
        self.eyeglasses = 0
        self.hat = 0
        self.mask = 0
        seed(2)
        with open(csvfile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            print('inside csvfile')
            cont = 0
            max_samples = samples
            #max_samples = -1
            for row in reader:
                if cont % 30000 == 0:
                    print(cont)
                file = row['Filename']

                if os.path.exists(path + "train/" + file) == True:
                    filename = path + "train/" + file
                else:
                    filename = path + "test/" + file
                
                if int(row['Mustache']) == 1 or int(row['5_o_Clock_Shadow']) == 1 or int(row['Goatee']) == 1:
                    beard = 1
                    self.beard += 1
                else:
                    beard = 0

                if int(row['Eyeglasses']) == 1 or int(row['No_Eyewear']) == -1:
                    eyeglasses = 1
                    self.eyeglasses += 1
                else:
                    eyeglasses = 0

                if int(row['Wearing_Hat'])  == 1:
                    hat = 1
                    self.hat += 1
                else:
                    hat = 0
                #if theres not occlusions in sample --> SKIP
                if hat == 0 and eyeglasses == 0 and beard == 0:
                    pass
                else:
                    self.samples.append((filename, (beard, eyeglasses, hat)))
                    cont += 1
                if cont == max_samples:
                    break

            self.transform = transform
            
    def __len__(self):        
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx][0]
        img = cv2.imread(sample)

        if self.transform != None:
            img = self.transform(img)
           
        return img, torch.tensor(self.samples[idx][1][0]), torch.tensor(self.samples[idx][1][1]), torch.tensor(self.samples[idx][1][2])


