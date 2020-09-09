import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from utils import idx_to_class, TRANSFORM


def image_loader(image_name):
    """Load image into a tensor.

    Args:
        image_name (str):The path to the image.

    Returns:
        tensor: Image after being proccessed.
    """

    image = Image.open(image_name)
    image = TRANSFORM(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse inference parameters')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='The path for the trained model.')
    parser.add_argument('img_path', type=str, help='The input image.')
    args = parser.parse_args()

    img = image_loader(args.img_path)
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)

    model_dict = torch.load(args.model)
    print("model's accuracy is: {}".format(model_dict['acc']))

    model.load_state_dict(model_dict['net'])
    model.cuda().eval()

    with torch.no_grad():
        output = model(img.cuda())
        pred = output.data.cpu().numpy().argmax()

        print("The class is: {}".format(idx_to_class[int(pred)]))
