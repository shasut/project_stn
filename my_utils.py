
import os
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! (while using headless gpu cluser server)
import matplotlib.pyplot as plt
import numpy as np
import torch

def createDirectoryIfNotExits(modelSaveLocation):
    if not os.path.exists(modelSaveLocation):
        os.makedirs(modelSaveLocation)
    return modelSaveLocation

def plot_loss(trainLoss, valLoss, saveLocation='', plot_title="loss_plot_"):
    fig = plt.figure(figsize=(18, 6))
    plt.subplot(121)
    x = [x for x in range(0, len(valLoss))]

    plt.plot(x, trainLoss)
    plt.plot(x, valLoss)
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.subplot(122)
    plt.semilogy(x, trainLoss)
    plt.semilogy(x, valLoss)
    plt.ylabel('loss in Log')
    plt.xlabel('epoch')

    plt.legend(['train', 'validation'], loc='upper right')

    t = str(round(trainLoss[-1], 8))
    v = str(round(valLoss[-1], 8))
    plt.suptitle(plot_title + '_t_' + t + '_v_' + v, fontsize=10)

    fig.savefig(saveLocation + plot_title + '_loss_plot.png')
    plt.close(fig)




def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction).cpu() == which_class)
            probabilities.extend(stable_sigmoid(np.exp(output[:, which_class].cpu())))
    return [i.item() for i in actuals], [i.item() for i in probabilities]




def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


from sklearn.metrics import roc_curve, auc, confusion_matrix


def roc_multiclass(model, device, test_loader, num_classes, saveLocation, plot_title):
    lw = 2
    N_CLASS = num_classes
    f = plt.figure()
    for i in range(N_CLASS):
        actuals, class_probabilities = test_class_probabilities(model, device, test_loader, i)

        fpr, tpr, _ = roc_curve(actuals, class_probabilities)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=lw, label=plot_title+" ROC, digit {} (area = {})".format(i, round(roc_auc, 2)))

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    f.savefig(saveLocation + plot_title+"_roc.png", bbox_inches='tight', transparent=False)


import pandas as pd
import seaborn as sns
def plot_confusion_matrix(model, device, test_loader, saveLocation, plot_title):
    actuals, predictions = test_label_predictions(model, device, test_loader)
    arr = confusion_matrix(actuals, predictions)
    print("Confusion Matrix:\n", arr)
    class_names = ['0', '  1', ' 2', '  3', ' 4', '  5', ' 6', '7', ' 8', '9']
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Predicted Classes")
    plt.ylabel("Target Clasess")
    plt.savefig(saveLocation+ plot_title + "_confusion_matrix.png")

    # , cmap = 'BuGn'