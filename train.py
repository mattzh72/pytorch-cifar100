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
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, cache_intermediate_output, \
    compute_feature_map_kurtosis, compute_kernel_kurtosis, compute_all_conv2d_kernel_kurtoses, \
    compute_feature_map_covariance_distance_from_identity, feature_map_has_0_mean_1_var

def compute_kurtosis_sum(kurtosis_conv2d_feature_map):
    kurtosis_sum = 0
            
    for name,val in kurtosis_conv2d_feature_map.items():
        if name == name_of_first_conv and args.remove_first_conv2d_for_kurtosis_loss:
            pass
        else:
            kurtosis_sum = kurtosis_sum + val
    return kurtosis_sum

def compute_kurtosis_term(kurtosis_conv2d_feature_map, cross_entropy_loss):
    # Select log kurtosis penalty
    if args.subtract_log_kurtosis_loss:
        kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_feature_map)
        kurtosis_term = -1 * torch.log(kurtosis_sum)
    # Select inverse kurtosis penalty
    elif args.add_inverse_kurtosis_loss:
        kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_feature_map)
        kurtosis_term = 1/kurtosis_sum
    elif args.add_mse_kurtosis_loss != None:
        loss = nn.MSELoss()
        # If flag is set, discard first conv2d outputs
        if args.remove_first_conv2d_for_kurtosis_loss:
          kurtoses = [v for (k,v) in kurtosis_conv2d_feature_map.items() if k != name_of_first_conv]
        else:
          kurtoses = list(kurtosis_conv2d_feature_map.values())
        kurtoses = torch.stack(kurtoses)
        kurtosis_term = loss(kurtoses, torch.ones(kurtoses.shape[0]).cuda() * args.add_mse_kurtosis_loss)
    else:
        raise Error()
    return kurtosis_term * args.kurtosis_global_loss_multiplier

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
        
        kurtosis_conv2d_feature_map = {k:compute_feature_map_kurtosis(v) \
                              for (k,v) in conv2d_feature_map_cache.items()} 

        kurtosis_feature_map_metrics = {"train/kurtosis_conv2d_feature_maps/kurtosis_{}".format(k):v.item() \
                              for (k,v) in kurtosis_conv2d_feature_map.items()} 

      
        kurtosis_kernel = compute_all_conv2d_kernel_kurtoses(net)

        kurtosis_kernel_metrics = { "train/kurtosis_conv2d_kernels/kurtosis_{}".format(k):v \
                              for (k,v) in kurtosis_kernel.items()}

        # covariance_distance_identity_feature_map_metrics = { "train/covariance_dist_from_identity_bn_feature_maps/{}".format(k): \
        #                       compute_feature_map_covariance_distance_from_identity(v).item()  \
        #                       for (k,v) in batch_norm_feature_map_cache.items()} 

        covariance_distance_identity_feature_map_metrics = { "train/covariance_dist_from_identity_bn_feature_maps/{}".format(k): \
                              compute_feature_map_covariance_distance_from_identity(v).item()  \
                              for (k,v) in whitening_conv1x1_feature_map_cache.items()} 
        
        if args.no_learnable_params_bn and args.no_track_running_stats_bn:
              for (k,v) in batch_norm_feature_map_cache.items():
                  assert feature_map_has_0_mean_1_var(v)
                  
       
        cross_entropy_loss = loss_function(outputs, labels)
   
        pbar_descrip = f"CE Loss: {cross_entropy_loss.item()}"
        if args.kurtosis_loss:
            #kurtosis + cross entropy global loss
            kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_feature_map)
            kurtosis_term = compute_kurtosis_term(kurtosis_conv2d_feature_map, cross_entropy_loss)

            # constraint = int(cross_entropy_loss < kurtosis_term) * (kurtosis_term - cross_entropy_loss)

            loss = cross_entropy_loss + kurtosis_term # * 0.1 * constraint
            pbar_descrip += f"\tKurtosis Loss: {kurtosis_term.item()}"
        else:
            #cross entropy global loss
            loss = cross_entropy_loss

        loss.backward()
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

            wandb.log({**kurtosis_kernel_metrics, 
                      **kurtosis_feature_map_metrics, 
                      **covariance_distance_identity_feature_map_metrics,
                      **metrics,
                      **{'train/kurtosis_sum': kurtosis_sum.item() if args.kurtosis_loss else None,
                        'train/kurtosis_loss_term': kurtosis_term.item() if args.kurtosis_loss else None} }) 

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

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

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

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

        
        kurtosis_conv2d_feature_map = {k:compute_feature_map_kurtosis(v) \
                              for (k,v) in conv2d_feature_map_cache.items()} 

        kurtosis_feature_map_metrics = {"test/kurtosis_conv2d_feature_maps/kurtosis_{}".format(k):v.item() \
                              for (k,v) in kurtosis_conv2d_feature_map.items()} 

      
        kurtosis_kernel = compute_all_conv2d_kernel_kurtoses(net)

        kurtosis_kernel_metrics = { "test/kurtosis_conv2d_kernels/kurtosis_{}".format(k):v \
                              for (k,v) in kurtosis_kernel.items()}

        # covariance_distance_identity_feature_map_metrics = { "test/covariance_dist_from_identity_bn_feature_maps/{}".format(k): \
        #                       compute_feature_map_covariance_distance_from_identity(v).item()  \
        #                       for (k,v) in batch_norm_feature_map_cache.items()} 
        
        if args.no_learnable_params_bn and args.no_track_running_stats_bn:
              for (k,v) in batch_norm_feature_map_cache.items():
                  assert feature_map_has_0_mean_1_var(v)
            
        if args.wandb:
            wandb.log({**kurtosis_kernel_metrics, 
                      **kurtosis_feature_map_metrics,
                      **covariance_distance_identity_feature_map_metrics})

        cross_entropy_loss = loss_function(outputs, labels)
   
        if args.kurtosis_loss:
            #kurtosis + cross entropy global loss
            kurtosis_sum = compute_kurtosis_sum(kurtosis_conv2d_feature_map)
            kurtosis_term = compute_kurtosis_term(kurtosis_conv2d_feature_map, cross_entropy_loss)
                        
            loss = cross_entropy_loss + kurtosis_term
        else:
            #cross entropy global loss
            loss = cross_entropy_loss


        # test_loss += loss.item() 
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

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', test_correct.float() / len(cifar100_test_loader.dataset), epoch)

    return test_correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-no_learnable_params_bn', action='store_true', default=False, help='')
    parser.add_argument('-no_track_running_stats_bn', action='store_true', default=False, help='')
    parser.add_argument('-wandb', action='store_true', default=False, help='weights and biases is used')
    parser.add_argument('-verbose', action='store_true', default=False, help='whether or not to print')

    ##Kurtosis Args
    parser.add_argument('-kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-kurtosis_global_loss_multiplier', type=float, default=None, help='hyperparameter multiplier for kurtosis loss term in global loss')
    parser.add_argument('-remove_first_conv2d_for_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-subtract_log_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_inverse_kurtosis_loss', action='store_true', default=False, help='')
    parser.add_argument('-add_mse_kurtosis_loss', type=float, default=None, help='')
    parser.add_argument('-checkpoint', action='store_true', default=False, help='store checkpoints')
    
    
    #Whitening Args
    parser.add_argument('-post_whitening', action='store_true', default=False, help='')
    parser.add_argument('-pre_whitening', action='store_true', default=False, help='')
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
    conv2d_feature_map_cache = {}
    name_of_first_conv = None
    
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            if 'conv' in name:
                layer.register_forward_hook(cache_intermediate_output(name, conv2d_feature_map_cache))
                if name_of_first_conv is None:
                    name_of_first_conv = name
    

    whitening_conv1x1_feature_map_cache = {}
    whitening_layers = {}
    
    
    for name, layer in net.named_modules():
        if 'whitening' in name:
            layer.register_forward_hook(cache_intermediate_output(name, conv2d_feature_map_cache))
            whitening_layers[name] = layer
    
    whitening_optimizers = {}
    for name, layer in whitening_layers.items():
        whitening_optimizers[name] = optim.SGD(layer.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            
    if args.wandb:
        import wandb

    #set up intermediate layer feature map caching
    batch_norm_feature_map_cache = {}
    for name, layer in net.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
          # moe
          layer.register_forward_hook(cache_intermediate_output(name, batch_norm_feature_map_cache))
    
    

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

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if args.checkpoint:
      if not os.path.exists(checkpoint_path):
          os.makedirs(checkpoint_path)
      checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    if args.wandb:
        wandb.init(
          project="Mehdi", 
          entity="tejm",
          config={
            "epochs": settings.EPOCH,
            "batch_size": args.b,
            "lr": args.lr,
            "warm": args.warm,
            "bn_learnable_affine_params": not args.no_learnable_params_bn,
            "bn_track_running_stats": not args.no_track_running_stats_bn,
            "net": args.net,
            "kurtosis_loss": args.kurtosis_loss,
            "kurtosis_global_loss_multiplier": args.kurtosis_global_loss_multiplier,
            'remove_first_conv2d_for_kurtosis_loss': None if not args.kurtosis_loss \
                           else args.remove_first_conv2d_for_kurtosis_loss,
            'add_inverse_kurtosis_loss': None if not args.kurtosis_loss \
                           else args.add_inverse_kurtosis_loss,
            'subtract_log_kurtosis_loss': None if not args.kurtosis_loss \
                           else args.subtract_log_kurtosis_loss,
            'add_mse_kurtosis_loss': args.add_mse_kurtosis_loss
          }
        )

        wandb.run.summary["best_accuracy"] = 0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)

        acc = eval_training(epoch)
        if args.wandb:
            wandb.run.summary["best_accuracy"] = acc if acc > wandb.run.summary["best_accuracy"] else wandb.run.summary["best_accuracy"]

        #start to save best performance model after learning rate decay to 0.01
        if args.checkpoint and epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if args.checkpoint and not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()

    if args.wandb:

        # 🐝 Close your wandb run 
        wandb.finish()