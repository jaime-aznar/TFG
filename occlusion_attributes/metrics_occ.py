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

    preds[0] = preds[0].to('cpu')
    labels[0] = labels[0].to('cpu')
    cm = confusion_matrix(labels[0], preds[0].argmax(dim=1))

    plot_confusion_matrix(cm, classes=['No Eyeglasses','Eyeglasses'], char = 'eyeglasses.png')

    preds[1] = preds[1].to('cpu')
    labels[1] = labels[1].to('cpu')
    cm = confusion_matrix(labels[1], preds[1].argmax(dim=1))

    plot_confusion_matrix(cm, classes=['No Beard','Beard'], char = 'beard.png')

    preds[2] = preds[2].to('cpu')
    labels[2] = labels[2].to('cpu')
    cm = confusion_matrix(labels[2], preds[2].argmax(dim=1))

    plot_confusion_matrix(cm, classes=['No Hat','Hat'], char = 'hat.png')
    #plot_confusion_matrix(cm, classes=['[2-18]', '[18-30]', '[30-45]', '[45-60]', '[60-90]'], test_accuracy = acc)



def plot_confusion_matrix(cm, classes, char, normalize=False, title='', cmap=plt.cm.Blues):
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
    plt.savefig(char)
    plt.clf()

@torch.no_grad()
def get_all_preds(model, loader):
    model.to(device)
    eyes_tensor = torch.tensor([])
    eyes_tensor = eyes_tensor.to(device)
    beard_tensor = torch.tensor([])
    beard_tensor = beard_tensor.to(device)
    hat_tensor = torch.tensor([])
    hat_tensor = hat_tensor.to(device)

    eyes_label = torch.tensor([])
    eyes_label = eyes_label.to(device)
    beard_label = torch.tensor([])
    beard_label = beard_label.to(device)
    hat_label = torch.tensor([])
    hat_label = hat_label.to(device)

    for batch in loader:
        images, eyes, beard, hat= batch
        images = images.to(device)
        eyes = eyes.to(device)
        beard = beard.to(device)
        hat = hat.to(device)
        preds = model(images)

        eyes_tensor = torch.cat((eyes_tensor, preds['label_eyeglasses']), dim = 0)
        beard_tensor = torch.cat((beard_tensor, preds['label_beard']), dim = 0)
        hat_tensor = torch.cat((hat_tensor, preds['label_hat']), dim = 0)

        eyes_label = torch.cat((eyes_label, eyes), dim = 0)
        beard_label = torch.cat((beard_label, beard), dim = 0)
        hat_label = torch.cat((hat_label, hat), dim = 0)


    all_preds = [eyes_tensor, beard_tensor, hat_tensor]
    all_labels = [eyes_label, beard_label, hat_label]

    return all_preds, all_labels



def f1_score(model, data, target_names):
    dataloader = DataLoader(data, batch_size=16)
    preds, labels = get_all_preds(model, dataloader)

    preds[0] = preds[0].to('cpu')
    labels[0] = labels[0].to('cpu')

    print(classification_report(y_true= labels[0], y_pred= preds[0].argmax(dim=1), target_names=['No Eyeglasses','Eyeglasses']))

    preds[1] = preds[1].to('cpu')
    labels[1] = labels[1].to('cpu')

    print(classification_report(y_true= labels[1], y_pred= preds[1].argmax(dim=1), target_names=['No Beard','Beard']))

    preds[2] = preds[2].to('cpu')
    labels[2] = labels[2].to('cpu')

    print(classification_report(y_true= labels[2], y_pred= preds[2].argmax(dim=1), target_names=['No Hat','Hat']))

