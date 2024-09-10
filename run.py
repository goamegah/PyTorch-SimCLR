import os
import argparse
import json

import torch
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch.utils.data import DataLoader

from simclr.data.simclr_dataset import SimclrDataset
from simclr.data.data_loader import get_dataloaders_mnist

from simclr.modules.simclr import SimCLR
from simclr.models.resnet import ResNet18, BasicBlock

from simclr.utils.config import load_checkpoint
from simclr.utils.train_v2 import train, eval

from simclr.utils.simclr_train_v2 import train_simclr
from simclr.utils.evaluate import set_all_seeds, compute_confusion_matrix
from simclr.utils.plotting import show_examples, plot_confusion_matrix

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.getenv('ROOT_DIR')

# Load configuration from JSON file
with open(f'{ROOT_DIR}/simclr/config/config_ckpt.json', 'r') as f:
    config = json.load(f)

model_urls = config['model_urls']
CHECKPOINT_PATH = model_urls['simclr_resnet18_pretrained_ckpt']
CHECKPOINT_FINETUNED_PATH = model_urls['simclr_resnet18_finetuned_ckpt']

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

##########################
# SETTINGS
##########################

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-d', '--data',
                    metavar='DIR',
                    default='./datasets',
                    help='path to dataset')

parser.add_argument('-m', '--mode',
                    default='train',
                    help='Whether to perform training or evaluation.',
                    choices=['train', 'eval'])

parser.add_argument('-tm', '--train-mode',
                    default='finetune',
                    help='The train mode controls different objectives and trainable components.',
                    choices=['pretrain', 'finetune'])

parser.add_argument('-dn', '--dataset-name',
                    default='mnist',
                    help='dataset name',
                    choices=['mnist'])

parser.add_argument('-a', '--arch',
                    metavar='ARCH',
                    default='simclr_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

parser.add_argument('-j', '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-te', '--train-epochs',
                    default=100,
                    type=int,
                    metavar='TE',
                    help='number of total epochs to run train')

parser.add_argument('-ee', '--eval-epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run test')

parser.add_argument('-b', '--batch-size',
                    default=10,
                    type=int,
                    metavar='B',
                    help='train batch size.')

parser.add_argument('--eval-batch-size',
                    default=256,
                    help='The test batch size to use during evaluation '
                         'mode (must be less or equal min(train, valid, test) size.')

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

parser.add_argument('--model-path',
                    default='./artefacts/simclr_finetuned.pth',
                    type=str,
                    help='model path for evaluation')

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

parser.add_argument('--out_dim',
                    default=128,
                    type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--log-every-n-steps',
                    default=10,
                    type=int,
                    help='Log every n steps')

parser.add_argument('--temperature',
                    default=0.07,
                    type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--n-views',
                    default=2,
                    type=int,
                    metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu-index',
                    default=0,
                    type=int,
                    help='Gpu index.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    set_all_seeds(args.seed)

    if args.mode == "eval":
        _, _, test_dl = get_dataloaders_mnist(batch_size=10, eval_batch_size=256,
                                              num_workers=8, train_size=100)

        model = ResNet18(num_layers=18,
                         block=BasicBlock,
                         num_classes=10,
                         grayscale=True)

        ckpt = load_checkpoint(model_path=CHECKPOINT_FINETUNED_PATH, device=args.device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=args.device)
        
        eval(model, test_dl, args)
        
    else:  # train

        if args.train_mode == 'pretrain':

            model = SimCLR(projection_dim=args.out_dim)

            simclr_ds = SimclrDataset(root=args.data)

            train_ds = simclr_ds.get_dataset(dataset_name=args.dataset_name, n_views=args.n_views)

            train_dl = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=None,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  drop_last=True)

            optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=len(train_dl),
                                                                   eta_min=0,
                                                                   last_epoch=-1)

            criterion = CrossEntropyLoss()
            #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
            with torch.cuda.device(args.gpu_index):
                train_simclr(model=model, optimizer=optimizer, scheduler=scheduler,
                             train_loader=train_dl, args=args, criterion=criterion)
                
        else: # fine-tuning train mode

            train_dl, valid_dl, test_dl = get_dataloaders_mnist(batch_size=10, eval_batch_size=256,
                                                                num_workers=8, train_size=100)
            ckpt = load_checkpoint(model_path=CHECKPOINT_PATH, device=args.device)

            state_dict = ckpt['state_dict']

            # remove prefix
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            model = ResNet18(num_layers=18,
                             block=BasicBlock,
                             num_classes=10,
                             grayscale=True)

            model.to(device=args.device)

            log = model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias']

            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
            criterion = CrossEntropyLoss()

            _, _, _ = train(model=model, optimizer=optimizer,
                            criterion=criterion, train_loader=train_dl,
                            valid_loader=test_dl, test_loader=test_dl,
                            args=args, name='simclr_resnet18_finetuned')

            model.cpu()
            # show_examples(model=model, data_loader=test_dl, results_dir='./figures')
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
                                           data_loader=test_dl,
                                           device=torch.device('cpu'))
            plot_confusion_matrix(mat,
                                  class_names=class_dict.values(),
                                  results_dir='./assets/plots')
            # plt.show()

if __name__ == "__main__":
    main()
