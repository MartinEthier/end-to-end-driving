import torch


class End2EndNet(torch.nn.Module):
    """
    """

    def __init__(self, encoder, decoder):
        super(End2EndNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, frames, prev_path):
        """
        frames shape: (batch_size, seq_len, 3, height, width)
        prev_path shape: (batch_size, seq_len, 3)
        """
        # Pass sequence through the encoder
        image_features = []
        for t in range(frames.shape[1]):
            image_features.append(self.encoder(frames[:, t])) # (batch_size, 125)
            
        # Combine image features into a sequence
        feature_sequence = torch.stack(image_features) # (seq_len, batch_size, 125)
        
        # Reorder prev_path dims and concat to image features
        path_features = prev_path.permute(1, 0, 2) # (seq_len, batch_size, 3)
        feature_sequence = torch.cat((feature_sequence, path_features), axis=2) # (seq_len, batch_size, 128)

        # Pass through decoder
        model_output = self.decoder(feature_sequence) # (batch_size, 3*future_steps)
        
        return model_output
