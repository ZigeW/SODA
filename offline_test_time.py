import argparse
import sys
sys.path.append('..')
from utils import *
from data_loader import load_test
import models.resnet_cifar as resnet_cifar
import models.resnet_imagenet as resnet_imagenet
from models.generator import ResnetGenerator
from offline_tta_ma import offline_tta as offline_tta_ma
from offline_tta_soda import offline_tta as offline_tta_soda

MODES = ['ma', 'da']
METHODS = ['soda']


def main(args):
    print(args)

    device = torch.device("cuda:" + str(args.workers[0]) if torch.cuda.is_available() and len(args.workers) > 0 else 'cpu')

    # Dataset
    if args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            dim_y = 10
        elif args.dataset == 'cifar100':
            dim_y = 100
        elif args.dataset.startswith('imagenet'):
            dim_y = 1000

    # test-time adaptation
    test_accs = 0.0
    for i_test, testdom in enumerate(args.testdoms):
        if args.seed is not None:
            deterministic_setting(args.seed)

        # Pretrained model
        if args.dataset.startswith('cifar'):
            model_arch = getattr(resnet_cifar, args.model)
            model = model_arch(args, num_classes=dim_y)
        elif args.dataset.startswith('imagenet'):
            model_arch = getattr(resnet_imagenet, args.model)
            model = model_arch()
        ckpt = torch.load(args.trained_model, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        if 1 < len(args.workers) < torch.cuda.device_count():
            model = torch.nn.DataParallel(model, device_ids=args.workers)
        print(f"Finished loading model {args.model}...", flush=True)

        # Data adaptation module
        if args.mode == 'da':
            adapt = ResnetGenerator(3, 3, args.ngf, norm_type='instance', act_type=args.activation,
                                    n_downsampling=args.n_downsample,
                                    n_blocks=args.n_resblocks, use_dropout=args.use_dropout,
                                    skip= args.mode != 'direct')
            adapt.to(device)
            print(f"Finished setup adaptation module...", flush=True)

            if args.zo:
                print(f"Using zeroth-order optimization...", flush=True)
            else:
                print(f"Using first-order optimization...", flush=True)
        else:
            adapt = None
            print(f"No use of data adaptation...")

        # Dataloader
        train_loader, test_loader = load_test(args.data_root, args.batch_size, args.dataset, testdom[:-1],
                                              int(testdom[-1]), args.train_ratio, workers=4)
        print(f"Finished loading test data {args.dataset} {testdom} with {len(train_loader.dataset)} training samples...", flush=True)

        print("Start test-time training...", flush=True)
        print(f"Mode {args.mode} method '{args.method}' on '{args.dataset} {args.testdoms[i_test]}':", flush=True)

        if args.mode == 'ma':
            val_accs = offline_tta_ma(args, model, train_loader, test_loader, device=device)
        elif args.mode == 'da':
            val_accs = offline_tta_soda(args, model, adapt, train_loader, test_loader, device=device)
        print(
            f"On dataset {args.dataset} {args.testdoms[i_test][:-1]} level {args.testdoms[i_test][-1]}, "
            f"method {args.method} final accuracy {val_accs[-1]:3f}", flush=True)
        test_accs += val_accs[-1]

        if args.save:
            save_dict = {'adapt': adapt.state_dict(),
                         'testdom': testdom,
                         'train_indices': train_loader.dataset.indices,
                         'args': args}
            save_name = f'checkpoints/{args.dataset}/{args.mode}_{args.method}_tr{args.train_ratio}_{args.testdoms[i_test]}_sd{args.seed}'
            save_name = unique_filename(save_name, '.pth.tar')
            torch.save(save_dict, save_name)
            print(f'Results saved to {save_name}.')

    print(f'Average accuracy on testdoms {(test_accs / len(args.testdoms))}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=MODES)  # test-time adaptation mode
    parser.add_argument("--method", type=str, choices=METHODS)  # method name
    parser.add_argument("--workers", type=int, nargs='+', default=[])  # number of workers

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument("--testdoms", type=str, nargs='+')  # corruption type and its severity level, e.g. snow1
    parser.add_argument("--trained_model", type=str, default=None)  # a trained model must be given
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save", action="store_true")  # save results
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=1)  # ratio of samples used for test-time training

    # test time training
    parser.add_argument("--batch_size", type=int, default=256)  # test batch size
    parser.add_argument("--optim", type=str, default='SGD')  # optimization method
    parser.add_argument("--lr", type=float, default=0.1)  # learning rate
    parser.add_argument("--steps", type=int, default=150)  # number of steps per batch
    parser.add_argument("--eval_interval", type=int, default=1)

    # DaTTA settings
    parser.add_argument("--ad_scale", type=float, default=1)  # the scale of adaptation
    parser.add_argument("--n_resblocks", type=int, default=0)
    parser.add_argument("--n_downsample", type=int, default=0)
    parser.add_argument("--use_dropout", type=boolstr, default=False)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--activation", type=str, default='hardswish', choices=['relu', 'hardswish'])
    parser.add_argument("--wdelta", type=float, default=0)  # weight of delta norm
    parser.add_argument("--rho", type=float, default=0.1)  # 1 - noise ratio
    parser.add_argument("--tau", type=float, default=0.9)   #confidence threshold
    parser.add_argument("--wclean", type=float, default=1e-4)  # weight of clean loss

    # ZO optimization
    parser.add_argument("--zo", type=boolstr, default=True)
    parser.add_argument('--q', default=10, type=int, metavar='N')  # query direction
    parser.add_argument('--mu', default=0.001, type=float, metavar='N')  # Smoothing Parameter
    parser.add_argument("--sigma", default=0.1, type=float)  # random direction vector sampling std

    args = parser.parse_args()

    main(args)

