# train.py
#!/usr/bin/env  python3

""" train network using pytorch

author baiyu
"""
import os
import sys
import argparse
import time
from datetime import datetime
from google.protobuf.descriptor import Error
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, cache_intermediate_output, \
    initialize_wandb, get_batch_norm_feature_map_cache, get_conv2d_feature_map_cache_and_name_of_first_conv, \
    toggle_grad_module, toggle_grad_params, get_statistics_from_layer_cache
    
from kurtosis_loss_utils import compute_kurtosis_sum, compute_kurtosis_term, \
    compute_feature_map_kurtosis, compute_all_conv2d_kernel_kurtoses, \
    compute_kernel_kurtosis

from whitening_loss_utils import compute_feature_map_covariance_distance_from_identity, \
      feature_map_has_0_mean_1_var, get_whitening_conv1x1_feature_map_cache, get_whitening_conv1x1s


def train(epoch):
    start = time.time()
    net.train()
    pbar = tqdm(cifar100_training_loader)
    for batch_index, (images, labels) in enumerate(pbar):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        
        outputs = net(images)
        
        # Get kurtosis statistics for feature maps of conv2d layers
        kurtosis_conv2d_fm, kurtosis_conv2d_metrics = {}, {}
        if args.kurtosis_loss:
          kurtosis_conv2d_fm, kurtosis_conv2d_metrics = get_statistics_from_layer_cache(
            conv2d_feature_map_cache, 
            statistics_func=compute_feature_map_kurtosis, 
            metrics_name_template="train/kurtosis_conv2d_fm/kurtosis_{}")

        # Get covariance statistics for conv1x1 layers feature map 
        covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = {}, {}
        if args.whitening_strength != None:
          covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = get_statistics_from_layer_cache(
            whitening_conv1x1_feature_map_cache, 
            statistics_func=compute_feature_map_covariance_distance_from_identity, 
            metrics_name_template="train/covariance_dist_from_identity_whitening_feature_maps/{}")
        
        if args.norm_checks and args.no_learnable_params_bn and args.no_track_running_stats_bn:
            for (k,v) in batch_norm_feature_map_cache.items():
                assert feature_map_has_0_mean_1_var(v)
      
        cross_entropy_loss = loss_function(outputs, labels)
        pbar_descrip = f"CE Loss: {cross_entropy_loss.item()}"
        if args.kurtosis_loss:
            #kurtosis + cross entropy global loss
            kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_fm, args)
            kurtosis_term = compute_kurtosis_term(kurtosis_conv2d_fm, cross_entropy_loss, args)

            loss = cross_entropy_loss + kurtosis_term 
            pbar_descrip += f"\tKurtosis Loss: {kurtosis_term.item()}"
            pbar_descrip += f"\tKurtosis Sum: {kurtosis_sum.item()}"
        else:
            #cross entropy global loss
            loss = cross_entropy_loss

        if args.whitening_strength != None:
          toggle_grad_module(net, False)
          # Conduct whitening loss first before CE loss
          for layer_name, distance in covariance_distance_identity_whitening_feature_map.items():
              whitening_loss_term = args.whitening_strength*distance
              toggle_grad_module(whitening_conv1x1s[layer_name], True)
              whitening_loss_term.backward(retain_graph = True)
              toggle_grad_module(whitening_conv1x1s[layer_name], False)
              
              pbar_descrip += f"\tWhitening Loss: {whitening_loss_term.item()}"
              pbar.set_description(pbar_descrip)

          toggle_grad_module(net, True)


        loss.backward(retain_graph = False)
        optimizer.step()
        
        pbar.set_description(pbar_descrip)

        if args.wandb:
            # Emit to wandb
            metrics = {
              "train/cross_entropy_loss": cross_entropy_loss.item(),
              # "train/epoch": batch_index * args.b + len(images) / len(cifar100_training_loader.dataset),
              "train/learning_rate": optimizer.param_groups[0]['lr'],
              "train/global_loss": loss.item()
            }

            wandb.log({**kurtosis_conv2d_metrics, 
                      # **covariance_distance_identity_feature_map_metrics,
                      **covariance_distance_identity_whitening_feature_map_metrics,
                      **metrics,
                      **{'train/kurtosis_sum': kurtosis_sum.item() if args.kurtosis_loss else None,
                        'train/kurtosis_loss_term': kurtosis_term.item() if args.kurtosis_loss else None} }) 

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        if args.verbose: 
          print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tGlobal Loss: {:0.4f}\tCE_loss: {:0.4f}\tKurtosis_loss: {:0.4f}\tLR: {:0.6f}'.format(
              loss.item(),
              cross_entropy_loss.item(),
              kurtosis_term.item() if args.kurtosis_loss else -1,
              optimizer.param_groups[0]['lr'],
              epoch=epoch,
              trained_samples=batch_index * args.b + len(images),
              total_samples=len(cifar100_training_loader.dataset)
          ))

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    if args.verbose:
      print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    test_correct = 0.0
    test_average_kurtosis_sum = 0
    test_average_kurtosis_term = 0
    test_average_cross_entropy_term = 0

    pbar = tqdm(cifar100_test_loader)
    for (images, labels) in pbar:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        # Get kurtosis statistics for feature maps of conv2d layers
        kurtosis_conv2d_fm, kurtosis_conv2d_metrics = get_statistics_from_layer_cache(
          conv2d_feature_map_cache, 
          statistics_func=compute_feature_map_kurtosis, 
          metrics_name_template="test/kurtosis_conv2d_fm/kurtosis_{}")

        # # Calculate conv2d kernel kurtoses
        # kurtosis_kernel = compute_all_conv2d_kernel_kurtoses(net)
        # # Get .items() for wandb metrics emitting
        # kurtosis_kernel_metrics = { "test/kurtosis_conv2d_kernels/kurtosis_{}".format(k):v \
        #                       for (k,v) in kurtosis_kernel.items()}

        covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = {}, {}
        if args.whitening_strength != None:
          covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = get_statistics_from_layer_cache(
            whitening_conv1x1_feature_map_cache, 
            statistics_func=compute_feature_map_covariance_distance_from_identity, 
            metrics_name_template="test/covariance_dist_from_identity_whitening_feature_maps/{}")
            
        if args.norm_checks and args.no_learnable_params_bn and args.no_track_running_stats_bn:
              for (k,v) in batch_norm_feature_map_cache.items():
                  assert feature_map_has_0_mean_1_var(v)
            
        if args.wandb:
            wandb.log({**kurtosis_conv2d_metrics, 
                      # **covariance_distance_identity_feature_map_metrics
                      **covariance_distance_identity_whitening_feature_map_metrics
                      })

        cross_entropy_loss = loss_function(outputs, labels)
   
        if args.kurtosis_loss:
            #kurtosis + cross entropy global loss
            kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_fm, args)
            kurtosis_term = compute_kurtosis_term(kurtosis_conv2d_fm, cross_entropy_loss, args)
                        
            loss = cross_entropy_loss + kurtosis_term
        else:
            #cross entropy global loss
            loss = cross_entropy_loss

        test_loss += loss.item() * len(images)/len(cifar100_test_loader.dataset)
        test_average_cross_entropy_term += cross_entropy_loss.item() * len(images)/len(cifar100_test_loader.dataset)

        if args.kurtosis_loss: 
            test_average_kurtosis_sum += kurtosis_sum.item() * len(images)/len(cifar100_test_loader.dataset)
            test_average_kurtosis_term += kurtosis_term.item() * len(images)/len(cifar100_test_loader.dataset)

        
        _, preds = outputs.max(1)
        test_correct += preds.eq(labels).sum()
    
    train_correct = 0.0
    for (images, labels) in tqdm(cifar100_training_loader):

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
      
        _, preds = outputs.max(1)
        train_correct += preds.eq(labels).sum()

    if args.verbose:
      finish = time.time()
      if args.gpu:
          print('GPU INFO.....')
          print(torch.cuda.memory_summary(), end='')
      print('Evaluating Network.....')
      print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
          epoch,
          # test_loss / len(cifar100_test_loader.dataset),
          test_loss,
          test_correct.float() / len(cifar100_test_loader.dataset),
          finish - start
      ))
      print()

    
    if args.wandb:
        wandb.log({"test/average_cross_entropy_loss": test_average_cross_entropy_term,
                  "test/average_global_loss": test_loss,
                  "test/test_accuracy": test_correct.float() / len(cifar100_test_loader.dataset),
                  "train/train_accuracy": train_correct.float() / len(cifar100_training_loader.dataset),
                  'test/average_kurtosis_sum': test_average_kurtosis_sum if args.kurtosis_loss else None,
                  'test/average_kurtosis_loss_term': test_average_kurtosis_term if args.kurtosis_loss else None})

    return test_correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-no_learnable_params_bn', action='store_true', default=False, help='')
    parser.add_argument('-no_track_running_stats_bn', action='store_true', default=False, help='')
    parser.add_argument('-wandb', action='store_true', default=False, help='weights and biases is used')
    parser.add_argument('-verbose', action='store_true', default=False, help='whether or not to print')
    parser.add_argument('-norm-checks', action='store_true', default=False, help='whether to check for correct normalization')

    #Kurtosis Args
    parser.add_argument('-kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-kurtosis_global_loss_multiplier', type=float, default=None, help='hyperparameter multiplier for kurtosis loss term in global loss')
    parser.add_argument('-remove_first_conv2d_for_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-subtract_log_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_inverse_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_mse_kurtosis_loss', type=float, default=None, help='')    
    
    #Whitening Args
    parser.add_argument('-post_whitening', action='store_true', default=False, help='')
    parser.add_argument('-pre_whitening', action='store_true', default=False, help='')
    parser.add_argument('-whitening_strength', type=float, default=None, help='')
    parser.add_argument('-switch_3x3conv2d_and_bn', action='store_true', default=False, help='')
    
    
    
    args = parser.parse_args()

    kurtosis_flags = [
      args.subtract_log_kurtosis_loss,
      args.add_inverse_kurtosis_loss,
      args.add_mse_kurtosis_loss != None
    ]

    if args.kurtosis_loss:
      assert sum(kurtosis_flags) == 1
    else:
      assert sum(kurtosis_flags) == 0

    assert bool(args.kurtosis_loss) == bool(args.kurtosis_global_loss_multiplier)    

    net = get_network(args)
    
    conv2d_feature_map_cache, args.name_of_first_conv \
                = get_conv2d_feature_map_cache_and_name_of_first_conv(net)

    if args.whitening_strength != None:
      whitening_conv1x1_feature_map_cache = get_whitening_conv1x1_feature_map_cache(net)
      whitening_conv1x1s, net_params_excluding_whitening_params = get_whitening_conv1x1s(net, get_excluded_layers=True)
            
    if args.wandb:
        import wandb

    #set up intermediate layer feature map caching
    batch_norm_feature_map_cache = get_batch_norm_feature_map_cache(net)
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=2,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=2,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    best_acc = 0.0

    if args.wandb:
        initialize_wandb(wandb, settings, args)
        
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)

        acc = eval_training(epoch)
        if args.wandb:
            wandb.run.summary["best_accuracy"] = acc if acc > wandb.run.summary["best_accuracy"] else wandb.run.summary["best_accuracy"]

    if args.wandb:

        # 🐝 Close your wandb run 
        wandb.finish()
