import os
import torch
import torch.nn as nn
import math
import numpy as np
import random

###################
#  Tools copied from https://github.com/changliu00/causal-semantic-generative-model
def boolstr(s: str) -> bool:
    # for argparse argument of type bool
    if isinstance(s, str):
        true_strings = {'1', 'true', 'True', 'T', 'yes', 'Yes', 'Y'}
        false_strings = {'0', 'false', 'False', 'F', 'no', 'No', 'N'}
        if s not in true_strings | false_strings:
            raise ValueError('Not a valid boolean string')
        return s in true_strings
    else:
        return bool(s)


def unique_filename(prefix: str="", suffix: str="", n_digits: int=2, count_start: int=0) -> str:
    fmt = "{:0" + str(n_digits) + "d}"
    if prefix and prefix[-1] not in {"/", "\\"}: prefix += "_"
    while True:
        filename = prefix + fmt.format(count_start) + suffix
        if not os.path.exists(filename): return filename
        else: count_start += 1


def is_same_tensor(ten1: torch.Tensor, ten2: torch.Tensor) -> bool:
    return (ten1 is ten2) or (
            type(ten1) == type(ten2) == torch.Tensor
            and torch.equal(ten1, ten2)) or (
            type(ten1) == type(ten2) == torch.Tensor
            and ten1.data_ptr() == ten2.data_ptr() and ten1.shape == ten2.shape)


class Averager:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, nrep = 1):
        self._val = val
        self._sum += val * nrep
        self._count += nrep
        self._avg = self._sum / self._count

    @property
    def val(self): return self._val
    @property
    def avg(self): return self._avg
    @property
    def sum(self): return self._sum
    @property
    def count(self): return self._count


# Test and save utilities
def acc_with_logits(logits: torch.LongTensor, y: torch.LongTensor, is_binary: bool) -> float:
    ypred = (logits > 0).long() if is_binary else logits.argmax(dim=-1)
    return (ypred == y).float().mean().item()


###########################
# training utilities
def warmup_learning_rate(curr_epoch, warm_epochs, batch_id, total_batches, warmup_from, warmup_to, optimizer):
    if curr_epoch <= warm_epochs:
        p = (batch_id + (curr_epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0., 1. / math.sqrt(m.in_features / 2.))
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def deterministic_setting(seed, deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FeatureQueue:
    def __init__(self, dim, length):
        self.length = length
        self.queue = torch.randn(length, dim)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def update(self, feat):

        batch_size = feat.shape[0]
        assert self.length % batch_size == 0  # for simplicity

        # replace the features at ptr (dequeue and enqueue)
        self.queue[self.ptr:self.ptr + batch_size] = feat
        self.ptr = (self.ptr + batch_size) % self.length  # move pointer
        if not self.full and self.ptr == 0:
            self.full = True

    def get(self, all=True):
        if all or self.full:
            return self.queue
        else:
            return self.queue[:self.ptr]

    def len(self):
        if self.full:
            return self.length
        else:
            return self.ptr


def scale_delta(delta, mag_in_scaled):
    bs = delta.shape[0]
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta[i, ci, :, :].detach().abs().max()
            delta[i, ci, :, :] = delta[i, ci, :, :].clone()\
                                 * np.minimum(1.0, mag_in_scaled[ci] / (l_inf_channel.cpu().numpy() + 1e-6))
    return delta


def clip_perturbed_image(x, x_tilda):
    # clip the input per channel
    for cii in range(3):
        x_tilda[:, cii, :, :] = x_tilda[:, cii, :, :].clone().clamp(x[:, cii, :, :].min(), x[:, cii, :, :].max())
    return x_tilda


class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, value):
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            return None
        root = self.heap[0]
        if len(self.heap) > 1:
            self.heap[0] = self.heap[-1]
            self.heap = self.heap[:-1]
            self._sift_down(0)
        else:
            self.heap = []
        return root

    def length(self):
        return len(self.heap)

    # def data(self):
    #     all_data = []
    #     for i in range(len(self.heap)):
    #         all_data.append(self.heap[i]['data'])
    #     all_data = torch.vstack(all_data)
    #     return all_data

    def index(self):
        all_index = []
        for i in range(len(self.heap)):
            all_index.append(self.heap[i]['index'])
        # all_index = torch.vstack(all_index)
        return all_index

    def _sift_up(self, index):
        parent_index = (index - 1) // 2
        if parent_index < 0:
            return
        if self.heap[parent_index]['value'] > self.heap[index]['value']:
            self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
            self._sift_up(parent_index)

    def _sift_down(self, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest_index = index
        if left_child_index < len(self.heap) and self.heap[left_child_index]['value'] < self.heap[smallest_index]['value']:
            smallest_index = left_child_index
        if right_child_index < len(self.heap) and self.heap[right_child_index]['value'] < self.heap[smallest_index]['value']:
            smallest_index = right_child_index
        if smallest_index != index:
            self.heap[index], self.heap[smallest_index] = self.heap[smallest_index], self.heap[index]
            self._sift_down(smallest_index)


def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    clone = clone._apply(lambda x: x)
    return clone


def detach_module(module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Detaches all parameters/buffers of a previously cloned module from its computational graph.
    Note: detach works in-place, so it does not return a copy.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key])


def set_dropout_to_eval(learner):
    for module in learner.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.eval()

def set_bn_to_eval(learner):
    for module in learner.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            module.eval()

