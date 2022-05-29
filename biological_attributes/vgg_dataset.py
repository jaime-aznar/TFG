import csv
from logging.config import fileConfig
import tarfile
from torch.utils.data import Dataset
import os
from fnmatch import fnmatch
import cv2
from torchvision import transforms
from random import randint
from random import seed

import matplotlib.pyplot as plt


class VggFace2(Dataset):
    
    def __init__(self, path, csvfile, transform = None, extract = 'Age'):
        print('Loading Dataset')
        self.samples = []
        self.n0 = 0
        self.n1 = 0
        self.n2 = 0

        with open(csvfile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            print('inside csvfile')
            cont = 0
            total_samples = 650000
            if extract == 'Age':
                seed(1)
            elif extract == 'Race':
                seed(1)
            else:
                seed(1)
            for row in reader:
                
                if cont == total_samples:
                    break
                vlue = randint(0, 10)
                if vlue % 2 == 0:
                    cont += 1
                    file = row['Filename']

                    if os.path.exists(path + "train/" + file) == True:
                        filename = path + "train/" + file
                    else:
                        filename = path + "test/" + file

                    if extract == 'Age':
                        if int(row['Young']) == 1:
                            tag = 0
                            self.n0 += 1
                        elif int(row['Middle_Aged']) != -1:
                            tag = 1
                            self.n1 += 1
                        else:
                            tag = 2
                            self.n2 += 1
                    elif extract == 'Gender':
                        if int(row['Male']) == 1:
                            tag  = 0
                            if self.n0 < (total_samples / 2):
                                self.n0 += 1
                            else:
                                cont -= 1  
                        else:
                            tag = 1
                            self.n1 += 1
                    else:
                        if int(row['Black']) != -1:
                            tag = 0
                            self.n0 += 1
                        elif int(row['Asian']) != -1:
                            tag = 1
                            self.n1 += 1
                        else:
                            tag = 2
                            self.n2 += 1
                    if extract == 'Gender':
                        if self.n0 < (total_samples / 2):
                            self.samples.append((filename, tag))
                    else:
                        self.samples.append((filename, tag))


            self.transform = transform

    def __len__(self):        
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx][0]
        img = cv2.imread(sample)

        if self.transform != None:
            img = self.transform(img)        

        return img, self.samples[idx][1]

