import torch


def finite_diff(batched_path):
    """
    """
    central_diff = (batched_path[:, 2:, :] - batched_path[:, :-2, :])/2
    back_diff = batched_path[:, -1:, :] - batched_path[:, -2:-1, :]
    forw_diff = batched_path[:, 1:2, :] - batched_path[:, 0:1, :]
    grads = torch.cat((forw_diff, central_diff, back_diff), 1)
    
    return grads

