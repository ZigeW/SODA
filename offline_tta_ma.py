import torch
import sys
import torch.nn.functional as F
import torch.nn as nn
sys.path.append('..')
from utils import Averager
from loss import *


def clean_sample_selection(model, test_loader, C, rho, tau, device):
    model.eval()

    clean_sets = {i:[] for i in range(C)}
    for x, _, idx in test_loader:
        x, idx = x.to(device), idx.to(device)
        output = model(x)
        prob = F.softmax(output, dim=-1)
        conf, pred = torch.max(prob, dim=-1)
        conf_high = conf > tau
        for i in torch.unique(pred):
            pred_i = pred == i
            clean_idx_i = torch.logical_and(conf_high, pred_i)
            if not (clean_idx_i == False).all():
                clean_sets[int(i)].append(torch.hstack([conf[clean_idx_i].unsqueeze(-1), idx[clean_idx_i].unsqueeze(-1),
                                                        pred[clean_idx_i].type(torch.int).unsqueeze(-1)]))

    k = len(test_loader.dataset) * rho / C
    clean_dataset = []
    for i in clean_sets.keys():
        if len(clean_sets[i]) > 0:
            clean_sets[i] = torch.vstack(clean_sets[i])
            if len(clean_sets[i]) > k:
                _, retain_set_i = torch.topk(clean_sets[i][:, 0], k=int(k))
                clean_sets[i] = clean_sets[i][retain_set_i]
            clean_dataset.append(clean_sets[i])
    assert len(clean_dataset) > 0
    clean_dataset = torch.vstack(clean_dataset)

    return clean_dataset


def train_clean(clean_x, clean_y, model, device):
    if clean_x.shape[0] == 0:
        return torch.tensor(0.0).to(device)
    logits = model(clean_x)
    loss_clean = nn.CrossEntropyLoss()(logits, clean_y)
    return loss_clean


def train_noisy(noisy_x, model, device):
    if noisy_x.shape[0] == 0:
        return torch.tensor(0.0).to(device)
    logits = model(noisy_x)
    loss_noisy = im_loss(logits, reduce=True)
    return loss_noisy


def offline_tta(args, model, train_loader, test_loader, device=None):
    parameters = get_model_parameters(model)
    model.train()

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr)

    # torch.autograd.set_detect_anomaly(True)
    accs = []
    with torch.no_grad():
        clean_sets = clean_sample_selection(model, train_loader, model.out_dim, args.rho, args.tau, device)
    print(f"number of clean data selected: {clean_sets.shape[0]}")

    for epoch in range(0, args.steps + 1):
        average = Averager()
        if epoch > 0:

            for i_bat, data_bat in enumerate(train_loader):
                x, y, idx = (data_bat[0].to(device), data_bat[1].to(device), data_bat[2].to(device))
                clean_mask = torch.isin(idx, clean_sets[:, 1])
                clean_set_mask = torch.isin(clean_sets[:, 1], idx)
                clean_x = x[clean_mask]
                noisy_x = x[~clean_mask]
                clean_y = clean_sets[clean_set_mask, 2].type(torch.long)

                optimizer.zero_grad()

                model.train()
                model.linear.eval()

                loss_clean = train_clean(clean_x, clean_y, model, device)
                loss_noisy = train_noisy(noisy_x, model, device)
                loss = args.wclean * loss_clean + loss_noisy
                # print(f"epoch {epoch}: loss {loss}, loss_tr {loss_tr}, delta_diff {delta_diff}, delta_norm {delta_norm}")

                loss.backward()
                optimizer.step()
            average.update(loss)

        if epoch % args.eval_interval == 0:
            avgr = Averager()
            model.eval()
            with torch.no_grad():
                for x, y, _ in train_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    ypred = logits.argmax(dim=-1)
                    avgr.update((ypred == y).float().mean().item(), nrep=len(y))
            acc = avgr.avg
            accs.append(acc)
            print(f"epoch {epoch:.1f}, loss = {average.avg}, acc = {acc:3f}.", flush=True)

    return accs