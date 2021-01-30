import torch


def finite_diff(batched_path):
    """
    """
    central_diff = (batched_path[:, 2:, :] - batched_path[:, :-2, :])/2
    back_diff = batched_path[:, -1:, :] - batched_path[:, -2:-1, :]
    forw_diff = batched_path[:, 1:2, :] - batched_path[:, 0:1, :]
    grads = torch.cat((forw_diff, central_diff, back_diff), 1)
    
    return grads

def grad_l1_loss(label, output):
    label_grads = finite_diff(label)
    output_reshaped = output.view(label.shape)
    output_grads = finite_diff(output_reshaped)
    loss = torch.nn.L1Loss()(label_grads, output_grads)
    
    return loss

def time_weighted_l1_loss(label, output):
    output_reshaped = output.view(label.shape)
    
    loss = torch.nn.L1Loss(reduction='none')(label, output_reshaped)
    
    t = torch.arange(label.shape[1])
    weights = (-torch.exp(-t/10) + 1.01).to(loss.device)
    
    tiled_weights = weights.unsqueeze(1).repeat(label.shape[0], 1, 3)
    
    weighted_loss = torch.mean(tiled_weights * loss)
    
    return weighted_loss
