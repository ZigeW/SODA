import torch
import torch.nn as nn


def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def get_model_parameters(model):
    params = []
    for nm, m in model.named_modules():
        if not nm.startswith('linear'):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
    return params


def im_loss(outputs_test, reduce=True):
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = Entropy(softmax_out)
    if reduce:
        entropy_loss = torch.mean(entropy_loss)

    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    # entropy_loss -= gentropy_loss
    im_loss = entropy_loss - gentropy_loss

    return im_loss

def entropy_loss(outputs_test, reduce=True):
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = Entropy(softmax_out)
    if reduce:
        entropy_loss = torch.mean(entropy_loss)
    return entropy_loss

