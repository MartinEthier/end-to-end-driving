import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import sys

from models.mixture_density_networks import MDN


class Decoder(nn.Module):
    """
    Wrapper class to initialize specified decoder.
    """
    def __init__(self, cfg, device):
        super(Decoder, self).__init__()
        self.decoder = getattr(sys.modules[__name__], cfg['model']['decoder']['name'])(cfg, device)
        
    def forward(self, *args):
        return self.decoder(*args)

class GRUDecoder(nn.Module):
    """
    Use GRU model to process sequence of image features and predict the final path using FC layers.
    """
    def __init__(self, cfg, device):
        super(GRUDecoder, self).__init__()
        dec_cfg = cfg['model']['decoder']['params']
        input_size = cfg['model']['encoder']['params']['enc_feat_len']
        future_steps = cfg['dataset']['future_steps']
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=dec_cfg['hidden_size'],
            num_layers=dec_cfg['num_layers'],
            dropout=dec_cfg['dropout_prob']
        )
        self.fc = nn.Linear(in_features=dec_cfg['hidden_size'], out_features=dec_cfg['fc_size'], bias=False)
        self.bn = nn.BatchNorm1d(num_features=dec_cfg['fc_size'])
        self.dropout = nn.Dropout(dec_cfg['dropout_prob'])
        self.fc_out = nn.Linear(in_features=dec_cfg['fc_size'], out_features=3*future_steps)

    def forward(self, X):
        # Input size: (seq_len, batch_size, enc_feat_len)
        _, X = self.gru(X) # Get last hidden state, (1, batch_size, hidden_size)
        X = self.dropout(torch.squeeze(X, 0))
        X = self.fc(X)
        X = self.bn(X)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        X = self.fc_out(X)
        return X

class FCDecoder(nn.Module):
    """
    Use FC layers to process single image features.
    """
    def __init__(self, cfg, device):
        super(FCDecoder, self).__init__()
        dec_cfg = cfg['model']['decoder']['params']
        input_size = cfg['model']['encoder']['params']['enc_feat_len']
        future_steps = cfg['dataset']['future_steps']
        
        self.fc = nn.Linear(in_features=input_size, out_features=dec_cfg['fc_size'], bias=False)
        self.bn = nn.BatchNorm1d(num_features=dec_cfg['fc_size'])
        self.dropout = nn.Dropout(dec_cfg['dropout_prob'])
        self.fc_out = nn.Linear(in_features=dec_cfg['fc_size'], out_features=3 * future_steps)
        if 'bias_init_path' in dec_cfg:
            # Init array is [x1,y1,z1,...,x30,y30,z30]
            np_arr = np.loadtxt(dec_cfg['bias_init_path'])
            torch_arr = torch.from_numpy(np_arr).to(torch.float)
            self.fc_out.bias = nn.Parameter(torch_arr)
        
    def forward(self, X):
        # Input size: (seq_len=1, batch_size, enc_feat_len)
        X = self.fc(torch.squeeze(X, 0))
        X = self.bn(X)
        X = nn.ReLU()(X)
        X = self.dropout(X)
        X = self.fc_out(X)
        return X

class GRUSeqModelDecoder(nn.Module):
    """
    Process single frame encoder features with GRU and predict the path using a sequence model approach.
    Might be useful to use a different optimizer/LR for the encoder and decoder.
    """
    def __init__(self, cfg, device):
        super(GRUSeqModelDecoder, self).__init__()
        dec_cfg = cfg['model']['decoder']['params']
        enc_feat_len = cfg['model']['encoder']['params']['enc_feat_len']
        future_steps = cfg['dataset']['future_steps']
        
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=enc_feat_len,
            num_layers=dec_cfg['num_layers'],
            dropout=dec_cfg['dropout_prob']
        )
        self.fc_out = nn.Linear(in_features=enc_feat_len, out_features=3)
        # if 'bias_init_path' in dec_cfg:
        #     # Init array is [x1,y1,z1,...,x30,y30,z30]
        #     np_arr = np.loadtxt(dec_cfg['bias_init_path'])
        #     torch_arr = torch.from_numpy(np_arr).to(torch.float)
        #     self.fc_out.bias = nn.Parameter(torch_arr)
        
    def forward(self, X, hidden):
        # X: (seq_len=1, batch_size, 3)
        # hidden: (1, batch_size, hidden_size)
        output, hidden = self.gru(X, hidden)
        # output: (seq_len=1, batch_size, hidden_size)
        # hidden: (1, batch_size, hidden_size)
        output = self.fc_out(output[0]).unsqueeze(0)
        # output: (1, batch_size, 3)
        return output, hidden

class GRUSeqModelMDNDecoder(nn.Module):
    """

    """
    def __init__(self, cfg, device):
        super(GRUSeqModelMDNDecoder, self).__init__()
        dec_cfg = cfg['model']['decoder']['params']
        enc_feat_len = cfg['model']['encoder']['params']['enc_feat_len']
        future_steps = cfg['dataset']['future_steps']
        
        self.gru = nn.GRU(
            input_size=output_size,
            hidden_size=enc_feat_len,
            num_layers=dec_cfg['num_layers'],
            dropout=dec_cfg['dropout_prob']
        )
        self.mdn = MDN(cfg, device)
        
    def forward(self, X, hidden):
        # X: (seq_len=1, batch_size, 3)
        # hidden: (1, batch_size, hidden_size)
        output, hidden = self.gru(X, hidden)
        # output: (seq_len=1, batch_size, hidden_size)
        # hidden: (1, batch_size, hidden_size)
        output = self.fc_out(output[0]).unsqueeze(0)
        # output: (1, batch_size, 3)
        return output, hidden



if __name__=="__main__":
    # Testing
    import yaml
    cfg_path = "/home/martin/projects/end-to-end-driving/configs/regnet_seq_model_mdn.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(cfg)

    mdn = MDN(cfg, "cpu")
    batch_size = 32
    feat_len = 256
    X = torch.randn((batch_size, feat_len))
    pi, mu, U = mdn(X)
    print(pi.shape)
    print(mu.shape)
    print(U.shape)
    y = torch.randn((batch_size, 3))
    loss = MDN.nll_loss(pi, mu, U, y)
    sample = MDN.sample(pi, mu, U)
    print(sample.shape)
