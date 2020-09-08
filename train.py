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


def load_data(root="/home/ali/spacesense/EuroSAT/2750/", batch_size=16):
    transform = transforms.Compose(
    [transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    # embed()

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    # embed()
    return train_data_loader, val_data_loader

# def test():


if __name__ == "__main__":
    train_loader, val_loader = load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet18(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    epochs = 150
    best_acc = 0

    # Training
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)        
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100 * correct / total

            print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
            epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), acc))

        # Testing
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = 100 * correct / total

                    print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
                    epoch, batch_idx, len(val_loader), test_loss/(batch_idx+1), acc))

            # Save best model
            if acc > best_acc:
                print('==> Saving model..')  
                state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                }
                torch.save(state, './checkpoints/best_model.pth')
                best_acc = acc

        scheduler.step()