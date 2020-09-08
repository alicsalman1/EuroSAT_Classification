import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from IPython import embed
import os
import argparse
import logging
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


idx_to_class = {0 : 'AnnualCrop',
                1 : 'Forest',
                2 : 'HerbaVeg',
                3 : 'Highway',
                4 : 'Industrial',
                5 : 'Pasture',
                6 : 'PermanentCrop',
                7 : 'Residential',
                8 : 'River',
                9 : 'SeaLake'}


def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"

    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in train_dl:
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()



def load_data(root="/home/ali/spacesense/EuroSAT/2750/", batch_size=32):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.3443, 0.3817, 0.4084), (0.2018, 0.1352, 0.1147))])

    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets, random_state=11)
    # embed()
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    # embed()

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    # mean, std = calc_normalization(val_data_loader)
    # embed()
    return train_data_loader, val_data_loader


def performance_report(model, val_loader):
    model_dict = torch.load("./checkpoints/best_model.pth")
    model.load_state_dict(model_dict['net'])
    model.cuda().eval()
    preds = []
    gt = []
    with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)

                    predicted = outputs.argmax(1).tolist()
                    gt += targets.tolist()
                    preds += predicted

    cl_report = classification_report(gt, preds, target_names=idx_to_class.values(), digits=3, output_dict=True)
    cl_report = pd.DataFrame(cl_report).transpose()
    print(cl_report)
    confusion = confusion_matrix(gt, preds)
    confusion = pd.DataFrame(confusion, index=idx_to_class.values(), columns=idx_to_class.values())
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
    plt.xticks(rotation=0) 
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cifar10 classification models.')
    parser.add_argument('--lr', default=0.1, help='Learning rate for trainig.')
    parser.add_argument('--batch_size', default=32, help='The batch size for training.')
    parser.add_argument('--log_file', type=str, default='training.log', help='A file to log the training and val losses and accuracies.')
    parser.add_argument('--data_dir', type=str, default="/home/ali/spacesense/EuroSAT/2750/", help='The directory where the dataset is stored.')
    parser.add_argument('--num_epochs', default=75, help='Number of total epochs.')
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file, 'a'))
    print = logger.info

    train_loader, val_loader = load_data(args.data_dir, args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet18(pretrained=True)
    # for param in model.parameters():
    #         param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    epochs = args.num_epochs
    best_acc = 0

    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)        
            
            outputs = model(inputs)
            # embed()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100 * correct / total

            print('train epoch : {} [{}/{}]| lr: {} | loss: {:.3f} | acc: {:.3f}'.format(
            epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'], train_loss/(batch_idx+1), acc))

        scheduler.step()

        # Testing
        if epoch % 2 == 0:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()

                    acc = 100 * test_correct / test_total

                    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
                    epoch, batch_idx, len(val_loader), test_loss/(batch_idx+1), acc))

            # Save best model
            if acc > best_acc:
                print('==> Saving best model...')  
                state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                }
                torch.save(state, './checkpoints/best_model.pth')
                best_acc = acc

    # Create performance report using the best model parameters.
    performance_report(model, val_loader)