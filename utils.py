""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    elif args.net == 'pytorch-resnet18':
        from models.resnet_pytorch import resnet18
        
        net = resnet18(num_classes = 100, 
                      bn_learnable_affine_params = not args.no_learnable_params_bn,
                      bn_track_running_stats = not args.no_track_running_stats_bn,
                      post_whitening = args.post_whitening,
                      pre_whitening = args.pre_whitening,
                      switch_3x3conv2d_and_bn = args.switch_3x3conv2d_and_bn)
        # print(net)
        # assert False
        

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

"""Matt"""
def cache_intermediate_output(name, cache):
  def hook(module, input, output):
      cache[name] = output.clone() 
  return hook

def compute_feature_map_covariance(feature_map):
    """
    feature_map: (b x c x h x w)
    Estimates the covariance matrix of the variables given by the 
    feature_map matrix, where rows are the variables and columns are the 
    observations.
    """
    assert len(feature_map.shape) == 4
    b,c,h,w = feature_map.shape
    
    feature_map_cbhw = torch.permute(feature_map, (1, 0, 2, 3)) # (c x b x h x w)
    feature_map_collapsed = feature_map_cbhw.reshape(c, b * h * w)
    
    covariance_matrix = torch.cov(feature_map_collapsed, correction = 1)
    assert len(covariance_matrix.shape) == 2 and covariance_matrix.shape[0] == c\
            and covariance_matrix.shape[1] == c

    return covariance_matrix


def feature_map_has_0_mean_1_var(feature_map):
    """
    feature_map: (b x c x h x w)
    """
    b,c,h,w = feature_map.shape
    mean_feature_map = torch.mean(feature_map, dim = (0, 2, 3))
    var_feature_map = torch.var(feature_map, dim = (0, 2, 3))

    # print(mean_feature_map, var_feature_map)

    return_check = torch.isclose(mean_feature_map, torch.zeros(c).cuda(), atol=1e-01).all() \
            and torch.isclose(var_feature_map, torch.ones(c).cuda(), atol=1e-01).all() or \
            torch.isnan(mean_feature_map).any() or torch.isnan(var_feature_map).any()

    if not return_check:
        print(torch.mean(mean_feature_map), torch.mean(var_feature_map), feature_map.shape)
    return return_check
def compute_feature_map_covariance_distance_from_identity(feature_map):
    """
    feature_map: (b x c x h x w)
    """
    b,c,h,w = feature_map.shape

    
    # assert feature_map_has_0_mean_1_var(feature_map)
    
    
    covariance_matrix = compute_feature_map_covariance(feature_map)
    distance = torch.linalg.norm(covariance_matrix - torch.eye(c).cuda())

    return distance



def compute_all_conv2d_kernel_kurtoses(net):
    kernels = {}
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d):
            params = list(layer.parameters())
            assert len(params) == 1
            kernels[name] = compute_kernel_kurtosis(params[0])
    # print([(k, compute_kernel_kurtosis(v)) for k,v in kernels.items()])

    return kernels

def compute_kernel_kurtosis(kernel):
  """
    kernel: (n x c x h x w)
  """
  assert len(kernel.shape) == 4

  (n, c, h, w) = kernel.shape
  kernel_hw_collapsed = kernel.reshape(n, c * h * w)
  mean = torch.mean(kernel_hw_collapsed, dim=0) # c * h * w
  assert len(mean.shape)==1 and mean.shape[0]== c * h * w
  diffs = torch.linalg.norm(kernel_hw_collapsed - mean, dim=-1) # n
  assert len(diffs.shape)==1 and diffs.shape[0]==n
  var = torch.mean(torch.pow(diffs, 2.0), dim=-1) # 1
  assert len(var.shape) == 0 #and var.shape[0] == 1, var.shape
  zscores = (diffs / torch.pow(var, 0.5)) # n
  assert len(zscores.shape) == 1 and zscores.shape[0] == n
  kurt = torch.mean(torch.pow(zscores, 4.0), dim=-1) # 1
  assert len(kurt.shape) == 0 #and kurt.shape[0] == 1, kurt
  


  # print(zscores.shape, channel_kurt.shape, var.shape, diffs.shape, mean.shape)

  return kurt
def compute_feature_map_kurtosis(feature_map):
  """
    feature_map: (b x c x h x w)
  """
  assert len(feature_map.shape) == 4

  (b, c, h, w) = feature_map.shape
  feature_map_hw_collapsed = feature_map.reshape(b, c, h * w)
  mean = torch.mean(feature_map_hw_collapsed, dim=1).unsqueeze(dim=1) # b x 1 x h*w
  diffs = torch.linalg.norm(feature_map_hw_collapsed - mean, dim=-1) # b x c
  var = torch.mean(torch.pow(diffs, 2.0), dim=1).unsqueeze(dim=1) # b x 1
  zscores = (diffs / torch.pow(var, 0.5)).squeeze() # b x c
  channel_kurt = torch.mean(torch.pow(zscores, 4.0), dim=1) # b
  


  # print(zscores.shape, channel_kurt.shape, var.shape, diffs.shape, mean.shape)

  return torch.mean(channel_kurt) 