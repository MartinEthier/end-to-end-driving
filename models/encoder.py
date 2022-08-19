import torch
from torch import nn
import timm


class Encoder(nn.Module):
    """
    Initializes a pretrained ImageNet feature extractor from timm.
    Input images have to be at least 224x224, scaled to be within 0 and 1, then normalized using:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    """
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        enc_cfg = cfg['model']['encoder']
        params = enc_cfg['params']
        
        # Initialize pretrained model with 0 classes to get pooled features
        self.timm_encoder = timm.create_model(
            enc_cfg['name'],
            pretrained=True,
            num_classes=0,
            in_chans=params['in_channels'],
            drop_rate=params['dropout_prob']
        )
        
        # Add an extra fc layer to map to desired feature vec length
        self.fc = nn.Linear(in_features=params['timm_feat_len'], out_features=params['enc_feat_len'], bias=False)
        self.bn = nn.BatchNorm1d(num_features=params['enc_feat_len'])
        self.dropout = nn.Dropout(params['dropout_prob'])

    def forward(self, X):
        X = self.timm_encoder(X) # (batch_size, feature_len)
        X = self.fc(X) # (batch_size, out_features)
        X = self.bn(X)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        return X

if __name__=="__main__":
    cfg = {
        "model": {
            "encoder": {
                "name": "resnet34",
                "params": {
                    "timm_feat_len": 512,
                    "enc_feat_len": 256
                }
            }
        }
    }
    model = Encoder(cfg) 
    output = model(torch.randn(4, 3, 224, 224))
    print(output.shape)
