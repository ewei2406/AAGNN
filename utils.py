import torch.nn.functional as F

def get_acc(prediction, truth, mask):
    """
    Cross entropy loss and accuracy
    """

    loss = F.cross_entropy(prediction[mask], target[mask]).item()

    acc = (prediction[mask] == target[mask]).sum()/len(target[mask])

    return loss
