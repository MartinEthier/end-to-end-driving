import torch
import sys


class Decoder(torch.nn.Module):
    """
    """
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        self.decoder = getattr(sys.modules[__name__], cfg['name'])(cfg['params'])
        
    def forward(self, X):
        return self.decoder(X)

class LSTMDecoder(torch.nn.Module):
    """
    Sequence model used to process the image features.
    """
    def __init__(self, cfg):
        super(LSTMDecoder, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=256, num_layers=1)
        self.fc1 = torch.nn.Linear(in_features=256, out_features=256, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=128, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(num_features=128)
        self.fc_out = torch.nn.Linear(in_features=128, out_features=40*3)

    def forward(self, X):
        _, (X, _) = self.lstm(X) # Get last hidden state
        X = self.fc1(torch.squeeze(X))
        X = self.bn1(X)
        X = torch.nn.LeakyReLU()(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = torch.nn.LeakyReLU()(X)
        X = self.fc_out(X)
        
        return X

class FCDecoder(torch.nn.Module):
    """
    """
    def __init__(self, cfg):
        super(FCDecoder, self).__init__()
        
    def forward(self, X):
        return None

