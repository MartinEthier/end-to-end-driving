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
        
        self.fc = torch.nn.Linear(in_features=cfg['feature_len'], out_features=125, bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=125)

    def forward(self, X):
        X = self.encoder(X) # (batch_size, feature_len, 1, 1)
        X = torch.reshape(X, X.shape[0:2]) # (batch_size, feature_len)
        X = self.fc(X) # (batch_size, 125)
        X = self.bn(X)
        X = torch.nn.LeakyReLU()(X)
        return X

