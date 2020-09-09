import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Subset

from utils import load_data, performance_report


parser = argparse.ArgumentParser(description='Parse Training parameters.')
parser.add_argument('--lr', default=0.1, help='The learning rate for trainig.')
parser.add_argument('--weight_decay', default=5e-4, help='The weight decay for trainig.')
parser.add_argument('--step_lr', default=25, help='The step size for learning rate scheduler.')
parser.add_argument('--batch_size', default=64,
                    help='The batch size for training.')
parser.add_argument('--log_file', type=str, default='training.log',
                    help='A file to log the training and val losses and accuracies.')
parser.add_argument('--data_dir', type=str, default="/home/ali/spacesense/EuroSAT/2750/",
                    help='The directory where the dataset is stored.')
parser.add_argument('--num_epochs', default=75, help='Number of total epochs.')
parser.add_argument('--eval_only', action='store_true',
                    help='To evaluate the model only')
parser.add_argument('--model', type=str, default="./checkpoints/best_model.pth",
                    help='The model to evaluate.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(args.log_file, 'a'))
print = logger.info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet18(pretrained=True)
# for param in model.parameters():
#         param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.1)

train_loader, val_loader = load_data(args.data_dir, args.batch_size)


def train(epoch):
    """Main training loop.

    Args:
        epoch (int): Current epoch.
    """

    model.train()
    train_loss = 0
    correct = 0
    total = 0
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

        print('train epoch : {} [{}/{}]| lr: {} | loss: {:.3f} | acc: {:.3f}'.format(
            epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'], train_loss/(batch_idx+1), acc))

    scheduler.step()


def test(epoch, best_acc):
    """Main testing loop.

    Args:
        epoch (int): Current epoch.
        best_acc (float): Current best accuracy.

    Returns:
        float: new best accuracy (can be not changed).
    """

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

    return best_acc


if __name__ == "__main__":
    epochs = args.num_epochs
    best_acc = 0
    test_every = 2

    if args.eval_only:
        model_dict = torch.load("./checkpoints/best_model.pth")
        model.load_state_dict(model_dict['net'])
        performance_report(model, val_loader)

    else:
        for epoch in range(epochs):
            # Training
            train(epoch)

            # Testing
            if epoch % test_every == 0:
                best_acc = test(epoch, best_acc)

        # Create performance report using the best model parameters.
        performance_report(model, val_loader)