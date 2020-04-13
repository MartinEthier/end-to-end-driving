import torch


from torchvision import models


# Init pretrained resnet
resnet50 = models.resnet152(pretrained=True)
print(resnet50)
# Remove last fully connecetd layer
modules=list(resnet50.children())[:-1]
resnet50=nn.Sequential(*modules)
for p in resnet50.parameters():
    p.requires_grad = False

    