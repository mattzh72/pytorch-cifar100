import torch
import torch.nn as nn
from google.protobuf.descriptor import Error

def compute_kurtosis_sum(kurtosis_cache, args):
    kurtosis_sum = 0
            
    for name, kurtosis in kurtosis_cache.items():
        if name == args.name_of_first_conv and args.remove_first_conv2d_for_kurtosis_loss:
            pass
        else:
            kurtosis_sum = kurtosis_sum + kurtosis
    return kurtosis_sum

def compute_kurtosis_term(kurtosis_cache, args):
    # Select subtract log kurtosis 
    if args.subtract_log_kurtosis_loss:
        kurtosis_sum = compute_kurtosis_sum(kurtosis_cache, args)
        kurtosis_term = -1 * torch.log(kurtosis_sum)
    # Select add log kurtosis ()
    elif args.add_log_kurtosis_loss:
        kurtosis_sum = compute_kurtosis_sum(kurtosis_cache, args)
        kurtosis_term = torch.log(kurtosis_sum)
    # Select inverse kurtosis penalty
    elif args.add_inverse_kurtosis_loss:
        kurtosis_sum = compute_kurtosis_sum(kurtosis_cache, args)
        kurtosis_term = 1/kurtosis_sum
    else:
        if args.add_mse_kurtosis_loss != None:
          loss = nn.MSELoss()
          target = args.add_mse_kurtosis_loss
        elif args.add_smoothl1_kurtosis_loss != None:
          loss = nn.SmoothL1Loss(beta=1e-3)
          target = args.add_smoothl1_kurtosis_loss

        # If flag is set, discard first conv2d outputs
        if args.remove_first_conv2d_for_kurtosis_loss:
          kurtoses = [v for (k,v) in kurtosis_cache.items() if k != args.name_of_first_conv]
        else:
          kurtoses = list(kurtosis_cache.values())
        kurtoses = torch.stack(kurtoses)
        kurtosis_term = loss(kurtoses, torch.ones(kurtoses.shape[0]).cuda() * target)

    return kurtosis_term * args.kurtosis_loss_multiplier


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
  # assert len(mean.shape)==1 and mean.shape[0]== c * h * w
  diffs = torch.linalg.norm(kernel_hw_collapsed - mean, dim=-1) # n
  # assert len(diffs.shape)==1 and diffs.shape[0]==n
  var = torch.mean(torch.pow(diffs, 2.0), dim=-1) # 1
  # assert len(var.shape) == 0 #and var.shape[0] == 1, var.shape
  zscores = (diffs / torch.pow(var, 0.5)) # n
  # assert len(zscores.shape) == 1 and zscores.shape[0] == n
  kurt = torch.mean(torch.pow(zscores, 4.0), dim=-1) # 1
  # assert len(kurt.shape) == 0 #and kurt.shape[0] == 1, kurt
  


  # print(zscores.shape, channel_kurt.shape, var.shape, diffs.shape, mean.shape)

  return kurt

def make_feature_map_0_mean_1_var(feature_map):
    """
    feature_map: (b x c x h x w)
    """
    b,c,h,w = feature_map.shape
    mean_feature_map = torch.mean(feature_map, dim = (0, 2, 3), keepdim=True)
    std_feature_map = torch.std(feature_map, dim = (0, 2, 3), keepdim=True)


    return (feature_map - mean_feature_map)/std_feature_map


def feature_map_has_0_mean_1_var(feature_map, atol=1e-1):
    """
    feature_map: (b x c x h x w)
    """
    b,c,h,w = feature_map.shape
    mean_feature_map = torch.mean(feature_map, dim = (0, 2, 3))
    var_feature_map = torch.var(feature_map, dim = (0, 2, 3))

    # print(mean_feature_map, var_feature_map)

    return_check = torch.isclose(mean_feature_map, torch.zeros(c).cuda(), atol=atol).all() \
            and torch.isclose(var_feature_map, torch.ones(c).cuda(), atol=atol).all() or \
            torch.isnan(mean_feature_map).any() or torch.isnan(var_feature_map).any()

    return return_check


def compute_global_kurtosis(feature_map, normalize_first=False, enable_safety_checks=False):
  """
    feature_map: (b x c x h x w)
  """
  if enable_safety_checks:
    assert len(feature_map.shape) == 4

  (b, c, h, w) = feature_map.shape
  # flatten into one dimension
  feature_map_collapsed = torch.flatten(feature_map)
  if enable_safety_checks:
    assert len(feature_map_collapsed.shape) == 1
    assert feature_map_collapsed.shape[0] == b * c * h * w

  # normalize if requested
  if normalize_first:
    m, std = torch.mean(feature_map_collapsed), torch.std(feature_map_collapsed)
    normalized_feature_map_collapsed = (feature_map_collapsed-m)/std
  else:
    normalized_feature_map_collapsed = feature_map_collapsed

  # print(torch.mean(normalized_feature_map_collapsed), torch.var(normalized_feature_map_collapsed))

  # get deviation
  mean = torch.mean(normalized_feature_map_collapsed) # 1
  diffs = normalized_feature_map_collapsed - mean # 1
  if enable_safety_checks:
    assert len(diffs.shape) == 1
    assert diffs.shape[0] == b * c * h * w

  # get second moment
  second_moment = torch.mean(torch.pow(diffs, 2)) # 1
  if enable_safety_checks:
    assert len(second_moment.shape) == 0

  # get fourth moment
  fourth_moment = torch.mean(torch.pow(diffs, 4)) # 1
  if enable_safety_checks:
    assert len(fourth_moment.shape) == 0

  # get kurtosis
  kurtosis_val = fourth_moment / (second_moment * second_moment)
  if enable_safety_checks:
    assert len(kurtosis_val.shape) == 0
    from scipy.stats import kurtosis
    scipy_kurtosis = kurtosis(normalized_feature_map_collapsed, fisher=False)
    
    import math
    assert math.isclose(scipy_kurtosis, kurtosis_val, rel_tol=1e-4)

  return kurtosis_val


def compute_feature_map_kurtosis(feature_map):
  """
    feature_map: (b x c x h x w)
  """
  assert len(feature_map.shape) == 4

  (b, c, h, w) = feature_map.shape
  feature_map_hw_collapsed = feature_map.reshape(b, c, h * w) # b x c x h * w
  channel_means = torch.mean(feature_map_hw_collapsed, dim=(0, 2)).unsqueeze(0).unsqueeze(2) # 1 x c x 1
  diffs = feature_map_hw_collapsed - channel_means # b x c x h * w

  # get second, fourth moment
  second_moment = torch.mean(torch.pow(diffs, 2), dim=(0, 2)) # c
  fourth_moment = torch.mean(torch.pow(diffs, 4), dim=(0, 2)) # c

  # get population kurtosis per channel
  return torch.div(fourth_moment, torch.pow(second_moment, 2))


