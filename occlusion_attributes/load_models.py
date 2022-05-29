import torch
import torch.nn as nn
from torchvision import transforms
import metrics_occ as m
from torch.utils.data import DataLoader
import multimodel as model
import vgg_occ_dataset as vgg

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



dataset = vgg.VggFace2(samples = 240000, path='../datasets/VGG-Face2/', csvfile='../datasets/MAAD_Face.csv', transform = transform)

print('Dataset loaded')

model = model.MultiCNN(False)
model.to('cuda')

model.load_state_dict(torch.load('best_3occ_model.pth'))

m.print_confussion_matrix(model, dataset)
#m.f1_score(model, dataset, target_names= ['Male', 'Female'])
