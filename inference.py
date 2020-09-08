from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models
import torch.nn as nn
from IPython import embed
import argparse


idx_to_class = {0 : 'AnnualCrop',
                1 : 'Forest',
                2 : 'HerbaceousVegetation',
                3 : 'Highway',
                4 : 'Industrial',
                5 : 'Pasture',
                6 : 'PermanentCrop',
                7 : 'Residential',
                8 : 'River',
                9 : 'SeaLake'}

def image_loader(image_name):
    """load image, returns cuda tensor"""
    transform = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize((0.3443, 0.3817, 0.4084), (0.2018, 0.1352, 0.1147))])
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cifar10 classification models.')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='The path for the trained model.')
    parser.add_argument('img_dir', type=str, help='The input image.')
    args = parser.parse_args()

    img = image_loader(args.img_dir)    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)

    model_dict = torch.load(args.model)
    print("model's accuracy is: {}".format(model_dict['acc']))
    model.load_state_dict(model_dict['net'])
    model.cuda().eval()

    output = model(img.cuda())
    pred = output.data.cpu().numpy().argmax()

    print("The class is: {}".format(idx_to_class[int(pred)]))
    print("Done!")
