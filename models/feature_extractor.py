import torch
import torchvision


class FeatureExtractor(torch.nn.Module):
    """
    Initializes a pretrained ImageNet feature extractor from torchvision.models
    Input images have to be at least 224x224, scaled to be within 0 and 1, then normalized using:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    """

    def __init__(self, model_name, input_size):
        super(FeatureExtractor, self).__init__()
        
        # Initialize pretrained model
        self.feature_extractor = getattr(torchvision.models, self.model_name)(pretrained=True)

        # Remove last fully connected layer and freeze weights
        modules = list(self.feature_extractor.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*modules)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, X):
        return self.feature_extractor(X)

