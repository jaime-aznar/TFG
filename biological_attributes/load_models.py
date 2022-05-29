import torch
import torch.nn as nn
from torchvision import models, transforms
import metrics as m
from torch.utils.data import DataLoader
import utk_dataset as mdl
#import vgg_dataset as vgg

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224)
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




#dataset = vgg.VggFace2(path='../datasets/VGG-Face2/', extract = 'Gender', csvfile='../datasets/MAAD_Face.csv', transform = transform)

print('Dataset loaded')

model = models.efficientnet_b0()
model.to('cuda')
last = len(model.classifier) - 1

num_features = model.classifier[last].in_features

model.classifier[last] = nn.Linear(num_features, 3)

model.load_state_dict(torch.load('accs/model_age_utk.pth'))


dataset = mdl.UTKFace(root_dir ='../../datasets/UTKFace/', transform = transform , extract= 0)

m.print_confussion_matrix(model, dataset)
