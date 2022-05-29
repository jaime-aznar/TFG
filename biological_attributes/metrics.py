import torch
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
device = 'cuda'

def print_confussion_matrix(model, data):
    conf = DataLoader(data, batch_size=128)
    preds, labels = get_all_preds(model, conf)
    preds = preds.to('cpu')
    labels = labels.to('cpu')
    cm = confusion_matrix(labels, preds.argmax(dim=1))
    
    #plot_confusion_matrix(cm, classes=['Hispanic','Black','Asian','Indian','Other'])
    plot_confusion_matrix(cm, classes=['Young', 'Middle-Aged', 'Old'])



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix Race', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
    plt.savefig('confussion_race.png')


@torch.no_grad()
def get_all_preds(model, loader):
    model.to(device)
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



def f1_score(model, data, target_names):
    dataloader = DataLoader(data, batch_size=128)
    preds, labels = get_all_preds(model, dataloader)
    preds = preds.to('cpu')
    labels = labels.to('cpu')
    print(classification_report(y_true= labels, y_pred= preds.argmax(dim=1), target_names=target_names))
