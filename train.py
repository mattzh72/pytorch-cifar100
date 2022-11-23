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
    toggle_grad_module, toggle_grad_params, get_statistics_from_layer_cache, get_fc_feature_map_cache
    
from kurtosis_loss_utils import compute_kurtosis_sum, compute_kurtosis_term, \
    compute_feature_map_kurtosis, compute_all_conv2d_kernel_kurtoses, \
    compute_kernel_kurtosis, compute_global_kurtosis

from whitening_loss_utils import compute_feature_map_covariance_distance_from_identity, \
      feature_map_has_0_mean_1_var, get_whitening_conv1x1_feature_map_cache, get_whitening_conv1x1s

from skewness_loss_utils import compute_skewness_term, compute_global_skewness, compute_channel_skewness


def train(epoch, kurtosis_loss_enabled=False):
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
        # TODO: If kurtosis loss is disabled, this will break - we expect to always be collecting 
        # kurtosis statistics and emitting to wandb
        # If kurtosis_loss is not enabled, then we will not have a statistics_func and this will break kurtosis_conv2d_fm below
        kurtosis_metrics = {}
        if args.kernel_kurtosis_loss: 
          # Calculate conv2d kernel kurtoses
          layer_kurtosis_map = compute_all_conv2d_kernel_kurtoses(net)
          # Get .items() for wandb metrics emitting
          kurtosis_metrics = { "train/kurtosis_conv2d_kernels/kurtosis_{}".format(k):v \
                                for (k,v) in layer_kurtosis_map.items()}
        else:
          # This is horrible, need to rework this logic
          if kurtosis_loss_enabled:
            if args.global_kurtosis_loss:
              statistics_func = compute_global_kurtosis
            elif args.fm_kurtosis_loss:
              statistics_func = compute_feature_map_kurtosis

            if args.fc_layer_kurtosis_only:
              cache = fc_feature_map_cache
              metrics_name_template = "train/kurtosis_fc_fm/kurtosis_{}"
            # default to conv2d featuremap
            else:
              cache = conv2d_feature_map_cache
              metrics_name_template = "train/kurtosis_conv2d_fm/kurtosis_{}"

            # Get kurtosis statistics for feature maps of conv2d layers
            layer_kurtosis_map, kurtosis_metrics = get_statistics_from_layer_cache(
              cache, 
              statistics_func=statistics_func, 
              metrics_name_template=metrics_name_template)

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

        loss = cross_entropy_loss
        if kurtosis_loss_enabled and epoch < args.kurtosis_warmup:  
            #kurtosis and cross entropy global loss
            kurtosis_sum = compute_kurtosis_sum(layer_kurtosis_map, args)
            kurtosis_term = compute_kurtosis_term(layer_kurtosis_map, args) 
            loss += kurtosis_term 
            total_num_layers = len(list(layer_kurtosis_map.keys()))
            pbar_descrip += f"\tKurtosis Loss: {kurtosis_term.item()}"
            pbar_descrip += f"\tAvg Kurtosis Per Layer: {kurtosis_sum.item()/total_num_layers}"

        skewness_metrics = {}
        if args.skewness_loss_enabled:
            # Get skewness statistics 
            statistics_func=compute_global_skewness
            if args.channel_skewness_loss:
              statistics_func=compute_channel_skewness

            layer_skewness_map, skewness_metrics = get_statistics_from_layer_cache(
                conv2d_feature_map_cache, 
                statistics_func=statistics_func, 
                metrics_name_template="train/skewness_conv2d_global/{}")    

            skewness_term = compute_skewness_term(layer_skewness_map, cross_entropy_loss, args)            
            skewness_metrics.update({'train/skewness_loss_term': skewness_term.item()})

            loss += skewness_term
            total_num_layers = len(list(layer_skewness_map.keys()))
            pbar_descrip += f"\tSkewness Loss: {skewness_term.item()}"

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
            
            # Include kurtosis metrics
            metrics.update({**kurtosis_metrics, 
                    **skewness_metrics,
                    **covariance_distance_identity_whitening_feature_map_metrics,
                    **{'train/kurtosis_sum': kurtosis_sum.item() if kurtosis_loss_enabled else None,
                      'train/kurtosis_loss_term': kurtosis_term.item() if kurtosis_loss_enabled else None} 
            })

            wandb.log(metrics) 

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        if args.verbose: 
          print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tGlobal Loss: {:0.4f}\tCE_loss: {:0.4f}\tKurtosis_loss: {:0.4f}\tLR: {:0.6f}'.format(
              loss.item(),
              cross_entropy_loss.item(),
              kurtosis_term.item() if kurtosis_loss_enabled and epoch < args.kurtosis_warmup else -1,
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
def eval_training(epoch=0, tb=True, kurtosis_loss_enabled=False):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    test_correct = 0.0
    test_average_kurtosis_sum = 0
    test_average_kurtosis_term = 0
    test_average_cross_entropy_term = 0

    pbar = tqdm(cifar100_test_loader)
    running_total=0
    for (images, labels) in pbar:
        running_total += len(images)
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)

        # TODO: Fix this, read above Todo in train()
        kurtosis_metrics = {}
        if args.kernel_kurtosis_loss: 
          # Calculate conv2d kernel kurtoses
          layer_kurtosis_map = compute_all_conv2d_kernel_kurtoses(net)
          # Get .items() for wandb metrics emitting
          kurtosis_metrics = { "train/kurtosis_conv2d_kernels/kurtosis_{}".format(k):v \
                                for (k,v) in layer_kurtosis_map.items()}
        else:
          # This is horrible, need to get rid of this logic
          if kurtosis_loss_enabled:
            if args.global_kurtosis_loss:
              statistics_func = compute_global_kurtosis
            elif args.fm_kurtosis_loss:
              statistics_func = compute_feature_map_kurtosis

            if args.fc_layer_kurtosis_only:
              cache = fc_feature_map_cache
              metrics_name_template = "test/kurtosis_fc_fm/kurtosis_{}"
            # default to conv2d featuremap
            else:
              cache = conv2d_feature_map_cache
              metrics_name_template = "test/kurtosis_conv2d_fm/kurtosis_{}"

            # Get kurtosis statistics for feature maps of conv2d layers
            layer_kurtosis_map, kurtosis_metrics = get_statistics_from_layer_cache(
              cache, 
              statistics_func=statistics_func, 
              metrics_name_template=metrics_name_template)

        covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = {}, {}
        if args.whitening_strength != None:
          covariance_distance_identity_whitening_feature_map, covariance_distance_identity_whitening_feature_map_metrics = get_statistics_from_layer_cache(
            whitening_conv1x1_feature_map_cache, 
            statistics_func=compute_feature_map_covariance_distance_from_identity, 
            metrics_name_template="test/covariance_dist_from_identity_whitening_feature_maps/{}")
            
        if args.norm_checks and args.no_learnable_params_bn and args.no_track_running_stats_bn:
              for (k,v) in batch_norm_feature_map_cache.items():
                  assert feature_map_has_0_mean_1_var(v)

        cross_entropy_loss = loss_function(outputs, labels)
        #kurtosis and cross entropy global loss

        loss = cross_entropy_loss
        if kurtosis_loss_enabled and epoch < args.kurtosis_warmup:  
            kurtosis_sum = compute_kurtosis_sum(layer_kurtosis_map, args)
            kurtosis_term = compute_kurtosis_term(layer_kurtosis_map, args)                      
            loss += kurtosis_term

        # Get skewness statistics 
        skewness_metrics = {}
        if args.skewness_loss_enabled:
            statistics_func=compute_global_skewness
            if args.channel_skewness_loss:
              statistics_func=compute_channel_skewness

            layer_skewness_map, skewness_metrics = get_statistics_from_layer_cache(
              conv2d_feature_map_cache, 
              statistics_func=statistics_func, 
              metrics_name_template="test/skewness_conv2d_global/{}")
            skewness_term = compute_skewness_term(layer_skewness_map, cross_entropy_loss, args, eval_mode=True)
            loss += skewness_term

        test_loss += loss.item() * len(images)/len(cifar100_test_loader.dataset)
        test_average_cross_entropy_term += cross_entropy_loss.item() * len(images)/len(cifar100_test_loader.dataset)

        if kurtosis_loss_enabled: 
            test_average_kurtosis_sum += kurtosis_sum.item() * len(images)/len(cifar100_test_loader.dataset)
            test_average_kurtosis_term += kurtosis_term.item() * len(images)/len(cifar100_test_loader.dataset)

        
        _, preds = outputs.max(1)
        test_correct += preds.eq(labels).sum()
        pbar.set_description("Test Loss: {}\tAccuracy: {}".format(test_loss, test_correct.float()/ running_total))

    
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
        if kurtosis_loss_enabled:
          kurtosis_metrics.update({
            'test/average_kurtosis_sum': test_average_kurtosis_sum,
            'test/average_kurtosis_loss_term': test_average_kurtosis_term
          })

        #       if args.wandb and kurtosis_loss_enabled and epoch < args.kurtosis_warmup:
        # wandb.log({**kurtosis_metrics, 
        #           # **covariance_distance_identity_feature_map_metrics
        #           **covariance_distance_identity_whitening_feature_map_metrics
        #           })

        wandb.log({"test/average_cross_entropy_loss": test_average_cross_entropy_term,
                  "test/average_global_loss": test_loss,
                  "test/test_accuracy": test_correct.float() / len(cifar100_test_loader.dataset),
                  "train/train_accuracy": train_correct.float() / len(cifar100_training_loader.dataset),
                  **kurtosis_metrics,
                  **skewness_metrics})

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
    parser.add_argument('-fm_kurtosis_loss', action='store_true', default=False, help='enable feature map kurtosis loss')
    parser.add_argument('-global_kurtosis_loss', action='store_true', default=False, help='enable global kurtosis loss')
    parser.add_argument('-kernel_kurtosis_loss', action='store_true', default=False, help='enable kernel kurtosis loss')
    parser.add_argument('-kurtosis_loss_multiplier', type=float, default=None, help='hyperparameter multiplier for kurtosis loss term in global loss')
    parser.add_argument('-remove_first_conv2d_for_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-subtract_log_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_log_kurtosis_loss', action='store_true', default=False, help='this will minimize kurtosis')
    parser.add_argument('-add_inverse_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-fc_layer_kurtosis_only', action='store_true', default=False, help='only apply kurtosis loss to last layer')
    parser.add_argument('-add_mse_kurtosis_loss', type=float, default=None, help='')   
    parser.add_argument('-add_smoothl1_kurtosis_loss', type=float, default=None, help='')   
    parser.add_argument('-kurtosis_warmup', type=float, default=float('inf'), help='')
    
    #Skewness args
    parser.add_argument('-global_skewness_loss', action='store_true', default=False, help='')
    parser.add_argument('-channel_skewness_loss', action='store_true', default=False, help='')
    parser.add_argument('-remove_first_conv2d_for_skewness_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_mse_skewness_loss', type=float, default=None, help='')   
    parser.add_argument('-skewness_loss_multiplier', type=float, default=None, help='hyperparameter multiplier for skewness loss term')

    #Multiplier
    parser.add_argument('-adaptive_multiplier_ce_thresh', type=float, default=None, help='percentage threshold for statistical regularizer strength')
    parser.add_argument('-adaptive_multiplier_patience', type=int, default=None, help='')

    #Whitening Args
    parser.add_argument('-post_whitening', action='store_true', default=False, help='')
    parser.add_argument('-pre_whitening', action='store_true', default=False, help='')
    parser.add_argument('-whitening_strength', type=float, default=None, help='')
    parser.add_argument('-switch_3x3conv2d_and_bn', action='store_true', default=False, help='')
    
    
    
    args = parser.parse_args()
    ###############
    # SAFETY CHECKS
    ###############

    # infer skewness loss enabled
    args.skewness_loss_enabled = args.add_mse_skewness_loss is not None

    kurtosis_loss_enabled = False
    if args.fm_kurtosis_loss or args.global_kurtosis_loss or args.kernel_kurtosis_loss:
      kurtosis_loss_enabled = True

    kurtosis_flags = [
      args.subtract_log_kurtosis_loss,
      args.add_inverse_kurtosis_loss,
      args.add_mse_kurtosis_loss != None,
      args.add_smoothl1_kurtosis_loss != None,
      args.add_log_kurtosis_loss, 
    ]

    if kurtosis_loss_enabled:
      assert sum(kurtosis_flags) == 1
    else:
      assert sum(kurtosis_flags) == 0

    # assert bool(kurtosis_loss_enabled) == bool(args.kurtosis_global_loss_multiplier)    
    ##################
    # END SAFETY CHECKS
    ###################

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
    fc_feature_map_cache = get_fc_feature_map_cache(net)
    
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
        initialize_wandb(wandb, settings, args, kurtosis_loss_enabled=kurtosis_loss_enabled)
        
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch, kurtosis_loss_enabled=kurtosis_loss_enabled)

        acc = eval_training(epoch, kurtosis_loss_enabled=kurtosis_loss_enabled)
        if args.wandb:
            wandb.run.summary["best_accuracy"] = acc if acc > wandb.run.summary["best_accuracy"] else wandb.run.summary["best_accuracy"]

    if args.wandb:

        # 🐝 Close your wandb run 
        wandb.finish()
