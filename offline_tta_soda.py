import torch
import torch.nn.functional as F
import sys
from collections import OrderedDict
sys.path.append('..')
from utils import Averager, clip_perturbed_image
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
    # smallest_conf = {i: torch.min(clean_sets[i]).item() for i in clean_sets.keys()}
    # print(smallest_conf)
    clean_dataset = torch.vstack(clean_dataset)

    return clean_dataset


def train_clean(clean_x, clean_y, model, adapt, args, device, detach=False):
    if clean_x.shape[0] == 0:
        return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
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
        return torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
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


def offline_tta(args, model, adapt, train_loader, test_loader, device=None):
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

    # torch.autograd.set_detect_anomaly(True)
    accs = []
    with torch.no_grad():
        clean_sets = clean_sample_selection(model, train_loader, model.out_dim, args.rho, args.tau, device)
    print(f"number of clean data selected: {clean_sets.shape[0]}")

    for epoch in range(0, args.steps + 1):
        avg_clean = Averager()
        avg_noisy = Averager()
        avg_norm = Averager()
        if epoch > 0:
            for i_bat, data_bat in enumerate(train_loader):
                x, y, idx = (data_bat[0].to(device), data_bat[1].to(device), data_bat[2].to(device))
                clean_mask = torch.isin(idx, clean_sets[:, 1])
                clean_set_mask = torch.isin(clean_sets[:, 1], idx)
                clean_x = x[clean_mask]
                noisy_x = x[~clean_mask]
                clean_y = clean_sets[clean_set_mask, 2].type(torch.long)

                optimizer.zero_grad()

                adapt.train()
                model.eval()

                if not args.zo:
                    loss_clean, norm_clean = train_clean(clean_x, clean_y, model, adapt, args, device)
                    loss_noisy, norm_noisy = train_noisy(noisy_x, model, adapt, args, device)
                    # loss_clean += args.wdelta * norm_clean
                    # loss_noisy += args.wdelta * norm_noisy
                    loss_norm = (norm_clean * clean_x.shape[0] + norm_noisy * (noisy_x.shape[0])) / x.shape[0]
                    loss = args.wclean * loss_clean + loss_noisy + args.wdelta * loss_norm
                    loss.backward()
                else:
                    loss_cleans = []
                    loss_noisys = []

                    loss_tmp_clean, norm_clean = zoo(clean_x, clean_y, model, adapt, args, device, clean=True)
                    loss_tmp_noisy, norm_noisy = zoo(noisy_x, None, model, adapt, args, device, clean=False)

                    loss_cleans.append(loss_tmp_clean.detach().cpu().mean())
                    loss_noisys.append(loss_tmp_noisy.detach().cpu().mean())
                    loss_norm = (norm_clean*clean_x.shape[0]+norm_noisy*(noisy_x.shape[0]))/x.shape[0]

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
                avg_norm.update(loss_norm.detach())

        if epoch % args.eval_interval == 0:
            avgr = Averager()
            adapt.eval()
            model.eval()
            with torch.no_grad():
                for x, y, idx in train_loader:
                    x, y, idx = x.to(device), y.to(device), idx.to(device)
                    if epoch > 0:
                        delta = adapt(x)
                        x_tilda = x + args.ad_scale * delta
                        x_tilda = clip_perturbed_image(x, x_tilda)
                    else:
                        x_tilda = x
                    logits = model(x_tilda)
                    ypred = logits.argmax(dim=-1)
                    avgr.update((ypred == y).float().mean().item(), nrep=len(y))
            acc = avgr.avg
            accs.append(acc)
            print(f"epoch {epoch:.1f}, loss_clean = {avg_clean.avg}, acc = {acc:3f}, loss_noisy = {avg_noisy.avg:3f}, "
                  f"delta_norm = {avg_norm.avg}.", flush=True)

    return accs