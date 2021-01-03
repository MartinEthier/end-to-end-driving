import torch
import torchvision


class Encoder(torch.nn.Module):
    """
    Initializes a pretrained ImageNet feature extractor from torchvision.models
    Input images have to be at least 224x224, scaled to be within 0 and 1, then normalized using:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    In a paper, they got better performance by freezing the first 15
    blocks of a resnet-50, and fine-tuning the rest.
    """

    def __init__(self, cfg):
        super(Encoder, self).__init__()
        
        # Initialize pretrained model
        self.encoder = getattr(torchvision.models, cfg['name'])(pretrained=True)

        # Remove last fully connected layer
        modules = list(self.encoder.children())[:-1]
        self.encoder = torch.nn.Sequential(*modules)
        
        self.fc = torch.nn.Linear(in_features=512, out_features=125, bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=125)

    def replace_modules(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.Softplus())
            else:
                convert_relu_to_softplus(child)    

    def forward(self, X):
        X = self.encoder(X) # (batch_size, 512, 1, 1)
        X = self.fc(torch.squeeze(X)) # (batch_size, 125)
        X = self.bn(X)
        X = torch.nn.LeakyReLU()(X)
        return X

