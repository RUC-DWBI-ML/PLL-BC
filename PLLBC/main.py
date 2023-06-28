import os
import os.path
import torch
import torchvision.transforms as transforms
import argparse
import numpy as np
import utils.utils_loss as L
from utils.models import linear, mlp
from S_datasets.fashion import PFashion
from S_datasets.mnist import PMNIST
from S_datasets.kmnist import PKMNIST


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('-lr_decay', help='learning rate step decay size', type=int, default=50)
    parser.add_argument('-gamma', help='learning rate step decay rate', type=float, default=1.0)

    parser.add_argument('-wd', help='weight decay', type=float, default=1e-3)
    parser.add_argument('-bs', help='batch size', type=int, default=256)
    parser.add_argument('-ep', help='number of epochs', type=int, default=500)
    parser.add_argument('-dataset', help='specify a dataset', type=str, default='mnist',
                        choices=['mnist', 'fmnist', 'kmnist'], required=False)
    parser.add_argument('-model', help='model name', type=str, default='linear',
                        choices=['linear', 'mlp', 'convnet', 'resnet'], required=False)
    parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial',
                        choices=['binomial', 'pair'])
    parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.5)

    parser.add_argument('-nw', help='multi-process data loading', type=int, default=0, required=False)
    parser.add_argument('-dir', help='result save path', type=str, default='result/', required=False)
    parser.add_argument('-gpu', help='gpu id', default='0', type=str, required=False)
    parser.add_argument('-loss', help='loss func', default='csg_dw', type=str, required=False)
    parser.add_argument('-T', help='sharpen or smooth', type=float, default=1.0, required=False)
    parser.add_argument('-seed', help='seed', type=int, default=0, required=False)
    parser.add_argument('-opti', help='optimizer', type=str, default='sgd', required=False)
    args = parser.parse_args()
    print(args.__dict__)
    return args


def set_dir(args):
    save_dir = './' + args.dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


# calculate accuracy
def evaluate(loader, model, args):
    device = args.device
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, targets, _, _ in loader:
            images = images.to(device)
            output = model(images)
            _, pred = torch.max(output.data, 1)
            total += images.size(0)
            correct += (pred.cpu() == targets).sum().item()
        acc = 100 * float(correct) / float(total)
        return acc


def get_dataset(args):
    num_features = 0
    num_classes = 0
    train_dataset = None
    test_dataset = None
    if args.dataset == 'mnist':
        num_features = 28 * 28
        num_classes = 10
        num_training = 60000
        train_dataset = PMNIST(root='./mnist/',
                               download=True,
                               train=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
                               )
        test_dataset = PMNIST(root='./mnist/',
                              download=True,
                              train=False,
                              transform=transforms.Compose(
                                  [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                              partial_type=args.partial_type,
                              partial_rate=args.partial_rate
                              )

    elif args.dataset == 'fmnist':
        num_features = 28 * 28
        num_classes = 10
        num_training = 60000
        train_dataset = PFashion(root='./fashion/',
                                 download=True,
                                 train=True,
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                 partial_type=args.partial_type,
                                 partial_rate=args.partial_rate
                                 )
        test_dataset = PFashion(root='./fashion/',
                                download=True,
                                train=False,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
                                )

    elif args.dataset == 'kmnist':
        num_features = 28 * 28
        num_classes = 10
        num_training = 60000
        train_dataset = PKMNIST(root='./kmnist/',
                                download=True,
                                train=True,
                                transform=transforms.Compose(
                                    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
                                )
        test_dataset = PKMNIST(root='./kmnist/',
                               download=True,
                               train=False,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
                               )
    return train_dataset, test_dataset, num_features, num_classes


def get_dl(args):
    train_dataset, test_dataset, num_features, num_classes = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               num_workers=args.nw,
                                               drop_last=False,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs,
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)
    return train_loader, test_loader, num_features, num_classes


def get_model(args, num_features, num_classes):
    model = None
    if args.model == 'linear':
        model = linear(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        model = mlp(n_inputs=num_features, n_outputs=num_classes)
    if model is None:
        raise NotImplementedError
    return model


def get_cY(data):
    partial_labels = data
    compl_labels = torch.logical_not(partial_labels.bool())
    compl_labels = compl_labels.float() / compl_labels.sum(dim=1, keepdim=True)
    return compl_labels


def main():
    args = set_args()
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    args.device = device
    args.out = set_dir(args)
    lr_set = [args.lr]
    # step_size
    step_sizes = [args.lr_decay]
    start_epoch = 0
    end_epoch = args.ep
    set_seed(args.seed)
    train_loader, test_loader, num_features, num_classes = get_dl(args)
    for step_size in step_sizes:
        for lr in lr_set:
            partial_labels = train_loader.dataset.partial_targets
            compl_labels = get_cY(partial_labels)
            model = get_model(args, num_features, num_classes)
            model.to(device)
            optimizer = None
            if args.opti == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.wd, momentum=0.9)
            elif args.opti == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.wd)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.gamma)

            for epoch in range(start_epoch, end_epoch):
                # print ('training...')
                model.train()
                for i, (input, targets, partial_targets, indexes) in enumerate(train_loader):
                    input = input.to(device)
                    targets = targets.to(device)
                    output = model(input)
                    new_partial_label = None
                    new_compl_label = None
                    if args.loss == 'bc_max':
                        loss = L.bc_max_loss(output, partial_labels[indexes, :].to(device), args)
                    elif args.loss == "bc_wn":
                        loss = L.bc_wn_loss(output, partial_labels[indexes, :].to(device), args)
                    elif args.loss == "bc_avg":
                        loss = L.bc_avg_loss(output, partial_labels[indexes, :].to(device),
                                             compl_labels[indexes, :].to(device),
                                             args)
                    elif args.loss == "ce":
                        loss = L.ce_loss(output, targets)
                    if torch.isnan(loss):
                        print("nan: exit!")
                        return
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # update weights
                    if new_partial_label is not None:
                        partial_labels[indexes, :] = new_partial_label.detach().cpu()
                    if new_compl_label is not None:
                        compl_labels[indexes, :] = new_compl_label.detach().cpu()
                scheduler.step()
                train_acc = evaluate(train_loader, model, args)
                test_acc = evaluate(test_loader, model, args)

                res = {"te_acc": test_acc, "tr_acc": train_acc, "epoch": epoch + 1, "lr": str(scheduler.get_last_lr())}
                print(res)


if __name__ == '__main__':
    main()
