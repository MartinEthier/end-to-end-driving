import torch



class SequenceModel(torch.nn.Module):
    """
    Sequence model used to process the image features.
    """

    def __init__(self, cfg):
        super(SequenceModel, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size=512, hidden_size=cfg['hidden_size'], num_layers=1)
        self.fc1 = torch.nn.Linear(in_features=cfg['hidden_size'], out_features=256)
        self.fc2 = torch.nn.Linear(in_features=256, out_features=30*3)

    def forward(self, X):
        _, (X, _) = self.lstm(X) # Get last hidden state
        X = self.fc1(torch.squeeze(X))
        X = torch.nn.ReLU()(X)
        X = self.fc2(X)
        return X

