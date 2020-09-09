import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


idx_to_class = {0: 'AnnualCrop',
                1: 'Forest',
                2: 'HerbaVeg',
                3: 'Highway',
                4: 'Industrial',
                5: 'Pasture',
                6: 'PermanentCrop',
                7: 'Residential',
                8: 'River',
                9: 'SeaLake'}


TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.3443, 0.3817, 0.4084), (0.2018, 0.1352, 0.1147))])


def calc_normalization(train_dl: torch.utils.data.DataLoader):
    """Function to calculate the mean and srtandard deviation of the dataset.

    Args:
        train_dl (torch.utils.data.DataLoader): DataLoader of the dataset.

    Returns:
        mean (array): The mean of each of the RGB channels.

    """

    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in train_dl:
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()


def load_data(root, batch_size=32):
    """Load and split the dataset.

    Args:
        root (str): Path to the dataset folder.
        batch_size (int, optional): The batch size. Defaults to 32.

    Returns:
        DataLoader: Train and test data loaders.
    """

    dataset = torchvision.datasets.ImageFolder(root=root, transform=TRANSFORM)
    train_idx, val_idx = train_test_split(list(
        range(len(dataset))), test_size=0.2, stratify=dataset.targets, random_state=11)

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False)
    # mean, std = calc_normalization(val_data_loader)
    return train_data_loader, val_data_loader


def performance_report(model, val_loader):
    """Generates classification report and plots confusion matrix.

    Args:
        model (torch model): The trained model.
        val_loader (DataLoader): Test set data loader.
    """

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

    cl_report = classification_report(
        gt, preds, target_names=idx_to_class.values(), digits=3, output_dict=True)
    cl_report = pd.DataFrame(cl_report).transpose()
    print(cl_report)

    confusion = confusion_matrix(gt, preds)
    confusion = pd.DataFrame(
        confusion, index=idx_to_class.values(), columns=idx_to_class.values())

    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
    plt.xticks(rotation=0)
    plt.savefig("confusion_matrix.png")
    plt.show()
