import torch
from torchvision import models




class FeatureExtractor(torch.nn.Module):
    """
    Initializes a pretrained ImageNet feature extractor from torchvision.models
    """

    def __init__(self, model_name):
        super(FeatureExtractor, self).__init__()
        
        # Initialize pretrained model
        self.feature_extractor = getattr(models, self.model_name)(pretrained=True)

        # Remove last fully connected layer and freeze weights
        modules = list(self.feature_extractor.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*modules)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self):
        return None

