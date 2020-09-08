from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models
import torch.nn as nn
from IPython import embed



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
    img = image_loader("Residential_1016.jpg")    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)

    f = torch.load("checkpoints/best_model.pth")
    model.load_state_dict(f['net'])

    # model = torch.load("checkpoints/best_model.pth")
    model.cuda().eval()
    output = model(img.cuda())
    pred = output.data.cpu().numpy().argmax()
    # embed()
    print("The class idx is: {}".format(pred))
    print("Done!")
