import torch
import torch.nn.functional as F
import torch.utils.data as D
import sys
import numpy as np
from collections import OrderedDict
sys.path.append('..')
from utils import Averager, clip_perturbed_image, MinHeap
from loss import *


class CleanCache:
    def __init__(self, cache_size, num_classes):
        self.heaps = [MinHeap() for _ in range(num_classes)]
        self.heapsize = int(cache_size / num_classes)

    def update(self, confs, indices, class_index):
        for i in range(confs.shape[0]):
            item = {'value': confs[i], 'index': indices[i].item()}
            self.heaps[class_index].push(item)
        if self.heaps[class_index].length() > self.heapsize:
            for _ in range(self.heaps[class_index].length() - self.heapsize):
                self.heaps[class_index].pop()

    def all(self):
        all_indices = []
        all_labels = []
        for i, h in enumerate(self.heaps):
            all_indices.extend(h.index())
            all_labels.extend([i for _ in range(h.length())])
        return all_labels, all_indices


def train_clean(clean_x, clean_y, model, adapt, args, device, detach=False):
    if clean_x.shape[0] == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    delta = adapt(clean_x)
    x_tilda = clean_x + args.ad_scale * delta
    x_tilda = clip_perturbed_image(clean_x, x_tilda)
    if detach:
        x_tilda = x_tilda.detach()
    logits_tilda = model(x_tilda)
    loss_clean = nn.CrossEntropyLoss()(logits_tilda, clean_y)
    delta_norm = torch.linalg.norm(delta, ord=1, dim=(-2, -1)).mean()
    # loss_clean += args.wdelta * delta_norm
    return loss_clean, delta_norm


def train_noisy(noisy_x, model, adapt, args, device, detach=False):
    if noisy_x.shape[0] == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    delta = adapt(noisy_x)
    x_tilda = noisy_x + args.ad_scale * delta
    x_tilda = clip_perturbed_image(noisy_x, x_tilda)
    if detach:
        x_tilda = x_tilda.detach()
    logits = model(x_tilda)
    loss_noisy = im_loss(logits, reduce=True)
    delta_norm = torch.linalg.norm(delta, ord=1, dim=(-2, -1)).mean()
    # loss_noisy += args.wdelta * delta_norm
    return loss_noisy, delta_norm


def zoo(x, y, model, adapt, args, device, clean):
    # Forward Inference (Original)
    if clean:
        loss_0, norm = train_clean(x, y, model, adapt, args, device, detach=True)
    else:
        loss_0, norm = train_noisy(x, model, adapt, args, device, detach=True)

    # ZO gradient estimate
    with torch.no_grad():
        mu = torch.tensor(args.mu).to(device)
        q = torch.tensor(args.q).to(device)

        # ZO Gradient Estimation
        original_parameter = adapt.state_dict()
        u = OrderedDict()
        grad = OrderedDict()
        for name in original_parameter.keys():
            u[name] = torch.zeros(original_parameter[name].size()).to(device)
            grad[name] = torch.zeros(original_parameter[name].size()).to(device)
        for i in range(args.q):
            for name in u.keys():
                u[name] = torch.normal(0, args.sigma, size=u[name].size()).to(device)
                original_parameter[name] += mu * u[name]
            adapt.load_state_dict(original_parameter)

            if clean:
                loss_tmp, _ = train_clean(x, y, model, adapt, args, device, detach=True)
            else:
                loss_tmp, _ = train_noisy(x, model, adapt, args, device, detach=True)
            loss_diff = torch.tensor(loss_tmp - loss_0)
            if clean:
                loss_diff = args.wclean * loss_diff
            for name in u.keys():
                grad[name] += loss_diff / (mu * q) * u[name]
                original_parameter[name] -= mu * u[name]

        # return parameter
        adapt.load_state_dict(original_parameter)
        for name, p in adapt.named_parameters():
            if p.grad is None:
                p.grad = grad[name]
            else:
                p.grad += grad[name]
    return loss_tmp, norm



def online_tta(args, model, adapt, train_loader, test_loader, device=None):
    # black-box model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # optimizer
    parameters = adapt.parameters()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=1e-5)

    clean_sets = CleanCache(args.queue_size, model.out_dim)

    avgr = Averager()
    for i_bat, data_bat in enumerate(test_loader):
        x, y, idx = (data_bat[0].to(device), data_bat[1].to(device), data_bat[2].to(device))

        # select clean data
        model.eval()
        output = model(x)
        prob = F.softmax(output, dim=-1)
        conf, pred = torch.max(prob, dim=-1)
        pre_acc = (pred == y).float().mean().item()
        print(f"batch {i_bat}, before adaptation accuracy = {pre_acc}")
        conf_high = conf > args.tau
        for i in torch.unique(pred):
            pred_i = pred == i
            clean_idx_i = torch.logical_and(conf_high, pred_i)
            if not (clean_idx_i == False).all():
                clean_sets.update(conf[clean_idx_i], idx[clean_idx_i], i)

        clean_y, clean_idx = clean_sets.all()
        test_loader.dataset.targets[clean_idx] = clean_y
        sub_indices = clean_idx + (list(idx.cpu().numpy()))
        sub_indices = list(set(sub_indices))
        new_subset = D.Subset(test_loader.dataset, sub_indices)
        new_loader = D.DataLoader(new_subset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True, drop_last=False)
        print(f"batch {i_bat}, subset length {len(new_subset)}, clean data {len(clean_idx)}")

        for epoch in range(1, args.steps + 1):
            avg_clean = Averager()
            avg_noisy = Averager()
            avg_norm = Averager()
            for i_new, data_new in enumerate(new_loader):
                x_new, y_new, idx_new = (data_new[0].to(device), data_new[1].to(device), data_new[2].to(device))
                clean_mask = torch.isin(idx_new, torch.tensor(clean_idx).to(device))
                clean_x = x_new[clean_mask]
                noisy_x = x_new[~clean_mask]
                clean_y = y_new[clean_mask]
                optimizer.zero_grad()

                adapt.train()
                model.eval()

                if not args.zo:
                    loss_clean, norm_clean = train_clean(clean_x, clean_y, model, adapt, args, device)
                    loss_noisy, norm_noisy = train_noisy(noisy_x, model, adapt, args, device)
                    loss_norm = (norm_clean * clean_x.shape[0] + norm_noisy * (noisy_x.shape[0])) / x.shape[0]
                    loss = args.wclean * loss_clean + loss_noisy + args.wdelta * loss_norm
                    # loss = loss_noisy
                    loss.backward()
                else:
                    loss_cleans = []
                    loss_noisys = []

                    loss_tmp_clean, norm_clean = zoo(clean_x, clean_y, model, adapt, args, device, clean=True)
                    loss_tmp_noisy, norm_noisy = zoo(noisy_x, None, model, adapt, args, device, clean=False)

                    loss_cleans.append(loss_tmp_clean.detach().cpu().mean())
                    loss_noisys.append(loss_tmp_noisy.detach().cpu().mean())
                    loss_norm = (norm_clean * clean_x.shape[0] + norm_noisy * (noisy_x.shape[0])) / x.shape[0]

                if args.zo:
                    loss_record_clean = torch.mean(torch.tensor(loss_cleans))
                    loss_record_noisy = torch.mean(torch.tensor(loss_noisys))
                else:
                    loss_record_clean = loss_clean
                    loss_record_noisy = loss_noisy
                # print(f"epoch {epoch}: loss_record {loss_record:.4f}, delta_norm {delta_norm:.4f}", flush=True)

                optimizer.step()
                avg_clean.update(loss_record_clean)
                avg_noisy.update(loss_record_noisy)
                avg_norm.update((norm_clean*clean_x.shape[0]+norm_noisy*(noisy_x.shape[0]))/x_new.shape[0])
                # avg_norm.update((norm_noisy * (noisy_x.shape[0])) / x.shape[0])

            # evaluation
            if epoch % args.eval_interval == 0 or epoch == args.steps:
                adapt.eval()
                model.eval()
                with torch.no_grad():
                    delta = adapt(x)
                    x_tilda = x + args.ad_scale * delta
                    x_tilda = clip_perturbed_image(x, x_tilda)
                    logits = model(x_tilda)
                    ypred = logits.argmax(dim=-1)
                    acc = (ypred == y).float().mean().item()
                    print(f"batch {i_bat}, epoch {epoch} acc = {acc:3f}, loss_noisy = {avg_noisy.avg:3f}, loss_clean = {avg_clean.avg:3f}，"
                          f"delta_norm = {avg_norm.avg}.", flush=True)
                    if epoch == args.steps:
                        avgr.update(acc, nrep=len(y))
                        print(f"batch {i_bat}, final acc = {acc}")
        print(f"After batch {i_bat}, total accuract = {avgr.avg}")
    return avgr.avg