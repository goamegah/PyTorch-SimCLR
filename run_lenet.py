import os
import json
import argparse

from simclr.models.lenet import LeNet5
from simclr.utils.evaluate import set_all_seeds, compute_confusion_matrix, compute_accuracy
from simclr.utils.train_v2 import train, eval
from simclr.utils.plotting import show_examples, plot_confusion_matrix
from simclr.data.data_loader import get_dataloaders_mnist
from simclr.utils.config import load_checkpoint

import torch
from torch.nn import CrossEntropyLoss
import torch.backends.cudnn as cudnn
import torchvision

import matplotlib.pyplot as plt

model_names = ['LeNet5']

from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')

# Load configuration from JSON file
with open(f'{ROOT_DIR}/simclr/config/config_ckpt.json', 'r') as f:
    config = json.load(f)

model_urls = config['model_urls']
CHECKPOINT_PATH = model_urls['lenet5_train_0100']

##########################
# SETTINGS
##########################

parser = argparse.ArgumentParser(description='PyTorch LeNet')

parser.add_argument('-m', '--mode',
                    metavar='MODE',
                    default='train',
                    help='which mode use during running model',
                    choices=['train', 'eval'])

parser.add_argument('-data',
                    metavar='DIR',
                    default='./data',
                    help='path to dataset')

parser.add_argument('-dn', '--dataset-name',
                    default='mnist',
                    help='dataset name',
                    choices=['cifar10', 'mnist'])

parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='LeNet5',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: LeNet5)')

parser.add_argument('-j', '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('-te', '--train-epochs',
                    default=100,
                    type=int,
                    metavar='TE',
                    help='number of total epochs to run train')

parser.add_argument('-ee', '--eval-epochs',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run test')

parser.add_argument('-b', '--batch-size',
                    default=10,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-eval-batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='Cause of having mini dataset, must be between 5 and 10')

parser.add_argument('--lr', '--learning-rate',
                    default=0.0003,
                    type=float,
                    metavar='LR',
                    help='initial learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--n-classes',
                    default=10,
                    type=int,
                    help='number of classes (default: 10)')

parser.add_argument('--seed',
                    default=1,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--disable-cuda',
                    action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision',
                    action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--log-every-n-steps',
                    default=10,
                    type=int,
                    help='Log every n steps')


def main():
    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    # Other
    GRAYSCALE = True  # for MNIST dataset

    set_all_seeds(args.seed)
    # set_deterministic

    ##########################
    # ## MNIST DATASET
    ##########################

    resize_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(batch_size=10, eval_batch_size=256,
                                                                    num_workers=8, train_size=100,
                                                                    train_transforms=resize_transform,
                                                                    test_transforms=resize_transform)

    model = LeNet5(grayscale=True, num_classes=10)
    model.to(device=args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.1,
                                                           mode='max',
                                                           verbose=True)
    criterion = CrossEntropyLoss()

    if args.mode == 'train':

        # train arch
        _, _, _ = train(model=model,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        valid_loader=test_loader,
                        test_loader=test_loader,
                        args=args,
                        name='LeNet5',
                        criterion=criterion)

        model.cpu()
        # show_examples(model=model, data_loader=test_loader, results_dir='./figures')
        # plt.show()

        # class used for confusion matrix axis ticks
        class_dict = {0: '0',
                      1: '1',
                      2: '2',
                      3: '3',
                      4: '4',
                      5: '5',
                      6: '6',
                      7: '7',
                      8: '8',
                      9: '9'}

        # Confusion matrix for testing arch
        mat = compute_confusion_matrix(model=model,
                                       data_loader=test_loader,
                                       device=torch.device('cpu'))
        plot_confusion_matrix(mat,
                              class_names=class_dict.values(),
                              results_dir='./assets/plots')
        # plt.show()

    # eval
    else:

        ckpt = load_checkpoint(model_path=CHECKPOINT_PATH, device=args.device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=args.device)
        
        eval(model, test_loader, args)


# main program
if __name__ == '__main__':
    main()
