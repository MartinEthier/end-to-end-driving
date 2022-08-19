import torch


def finite_diff(batched_path):
    """
    Return gradient along the given path. Similar to numpy.gradient. Uses forward and 
    backward diff for the 2 end points and central diff for all the middle points.
    """
    central_diff = (batched_path[:, 2:, :] - batched_path[:, :-2, :])/2
    back_diff = batched_path[:, -1:, :] - batched_path[:, -2:-1, :]
    forw_diff = batched_path[:, 1:2, :] - batched_path[:, 0:1, :]
    grads = torch.cat((forw_diff, central_diff, back_diff), 1)
    
    return grads

def grad_l1_loss(label, output):
    """
    L1 loss between the gradient of the label path and the gradient of the
    predicted path. Idea is to force the predicted paths to be smoother and
    match the label path.
    """
    label_grads = finite_diff(label)
    output_reshaped = output.view(label.shape)
    output_grads = finite_diff(output_reshaped)
    loss = torch.nn.L1Loss()(label_grads, output_grads)
    
    return loss

def grad_l2_loss(label, output):
    """
    L2 loss between the gradient of the label path and the gradient of the
    predicted path. Idea is to force the predicted paths to be smoother and
    match the label path.
    """
    label_grads = finite_diff(label)
    output_reshaped = output.view(label.shape)
    output_grads = finite_diff(output_reshaped)
    loss = torch.nn.L2Loss()(label_grads, output_grads)
    
    return loss

# NOTE: Update this, currently not working as intended
def time_weighted_l1_loss(label, output):
    """
    L1 loss where the coordinates further along the path are weighted
    higher than at the start. Idea is to penalize the model more for the
    more difficult points at the end.
    """
    output_reshaped = output.view(label.shape)
    
    loss = torch.nn.L1Loss(reduction='none')(label, output_reshaped)
    # Need to reverse the exp and add as args to the function
    t = torch.arange(label.shape[1])
    weights = (-torch.exp(-t/10) + 1.01).to(loss.device)
    
    tiled_weights = weights.unsqueeze(1).repeat(label.shape[0], 1, 3)
    
    weighted_loss = torch.mean(tiled_weights * loss)
    
    return weighted_loss

def focal_r_loss(label, output, beta=10.0, gamma=1.0):
    """
    Focal loss adapted for regression using L1 loss. From
    https://arxiv.org/pdf/2102.09554.pdf.
    """
    # mean(sigmoid(abs(beta*l1))**gamma * l1)
    raise NotImplementedError
