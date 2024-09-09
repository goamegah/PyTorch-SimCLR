import logging
import os
import sys

import torch
import wandb  # Import wandb
from simclr.utils.config import save_config_file, save_checkpoint
from simclr.utils.evaluate import accuracy

def eval(model, 
         optimizer, 
         test_loader,
         args, 
         criterion=None):

    for epoch in range(args.epochs):
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\t"
              f"Top1 Test accuracy: {top1_accuracy.item()}\t"
              f"Top5 test accuracy: {top5_accuracy.item()}")
    

def train(model, optimizer,
          train_loader, valid_loader,
          test_loader, args,
          name='training',
          criterion=None):

    minibatch_loss_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = [], [], [], [], []

    # Initialize wandb
    wandb.init(project='simclr_fine_tuning', config=args)
    wandb.watch(model, log='all')

    # config logging file
    logging.basicConfig(filename=f'{name}.log', level=logging.DEBUG)
    # save config file
    # save_config_file('wandb_run', args)

    loss_hist_valid = [0] * args.train_epochs
    accuracy_hist_valid = [0] * args.train_epochs

    n_iter = 0

    for epoch in range(args.train_epochs):
        train_acc_epoch = 0
        train_loss_epoch = 0
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
            accuracy = is_correct.sum()

            train_loss_epoch += loss_value * y_batch.size(0)
            train_acc_epoch += accuracy
            minibatch_loss_list.append(loss_value)

            # Log metrics to wandb
            wandb.log({
                'train/loss': loss_value,
                'train/acc': accuracy,
                'global_step': n_iter
            })
            n_iter += 1  # Update global step counter

        train_acc_list.append(train_acc_epoch/len(train_loader.dataset))
        train_loss_list.append(train_acc_epoch/len(train_loader.dataset))

        if valid_loader is not None:
            valid_acc_epoch = 0
            valid_loss_epoch = 0
            # Evaluation at each epoch
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    pred = model(x_batch)
                    loss = criterion(pred, y_batch)
                    is_correct = (torch.argmax(input=pred, dim=1) == y_batch).float()
                    valid_loss_epoch += loss.item() * y_batch.size(0)
                    valid_acc_epoch += is_correct.sum()

                valid_loss_epoch /= len(valid_loader.dataset)
                valid_acc_epoch /= len(valid_loader.dataset)

            loss_hist_valid[epoch] = valid_loss_epoch
            accuracy_hist_valid[epoch] = valid_acc_epoch

            valid_acc_list.append(valid_acc_epoch)
            valid_loss_list.append(valid_loss_epoch)

            # Log validation metrics to wandb
            wandb.log({
                'valid/loss': loss_hist_valid[epoch],
                'valid/acc': accuracy_hist_valid[epoch],
                'epoch': epoch
            })

            print(f'Epoch {epoch +1} '
                  f'val_accuracy: {accuracy_hist_valid[epoch]: .4f} '
                  f'val_loss: {loss_hist_valid[epoch]: .4f} '
                  f'test data: {len(test_loader.dataset)} '
                  f'train data: {len(train_loader.dataset)} ')

    logging.info("Training has finished.")
    # save model checkpoints
    checkpoint_name = f'./artefacts/{args.arch}_finetuned_{args.train_epochs:04d}.pth.tar'
    save_checkpoint(state={'epoch': args.train_epochs,
                           'arch': args.arch,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           },
                    is_best=False,
                    filename=checkpoint_name)
    logging.info(f"Model checkpoint and metadata has been saved.")

    # Save model checkpoint to wandb
    wandb.save(checkpoint_name)

    return minibatch_loss_list, train_acc_list, valid_acc_list