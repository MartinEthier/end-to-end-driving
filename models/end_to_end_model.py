import torch


class End2EndNet(torch.nn.Module):
    """
    """

    def __init__(self, encoder, sequence_model):
        super(End2EndNet, self).__init__()
        self.encoder = encoder
        self.sequence_model = sequence_model

    def forward(self, X):
        """
        X shape: (batch_size, seq_len, 3, height, width)
        """
        # Pass sequence through the encoder
        image_features = []
        for t in range(X.shape[1]):
            features = self.encoder(X[:, t]) # (batch_size, 512, 1, 1)
            image_features.append(torch.squeeze(features))
            
        # Combine image features into a sequence
        feature_sequence = torch.stack(image_features) # (seq_len, batch_size, 512)

        # Pass through sequence model
        model_output = self.sequence_model(feature_sequence) # (batch_size, 3*future_steps)
        
        return model_output
