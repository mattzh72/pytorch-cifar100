import torch
import torch.nn as nn
from google.protobuf.descriptor import Error

def compute_kurtosis_sum(kurtosis_conv2d_feature_map, args):
    kurtosis_sum = 0
            
    for name,val in kurtosis_conv2d_feature_map.items():
        if name == args.name_of_first_conv and args.remove_first_conv2d_for_kurtosis_loss:
            pass
        else:
            kurtosis_sum = kurtosis_sum + val
    return kurtosis_sum

def compute_kurtosis_term(kurtosis_conv2d_feature_map, cross_entropy_loss, args):
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
          kurtoses = [v for (k,v) in kurtosis_conv2d_feature_map.items() if k != args.name_of_first_conv]
        else:
          kurtoses = list(kurtosis_conv2d_feature_map.values())
        kurtoses = torch.stack(kurtoses)
        kurtosis_term = loss(kurtoses, torch.ones(kurtoses.shape[0]).cuda() * args.add_mse_kurtosis_loss)
    else:
        raise Error()
    return kurtosis_term * args.kurtosis_global_loss_multiplier


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


def compute_feature_map_kurtosis(feature_map):
  """
    feature_map: (b x c x h x w)
  """
  assert len(feature_map.shape) == 4

  normalized_feature_map = make_feature_map_0_mean_1_var(feature_map)
  # assert feature_map_has_0_mean_1_var(normalized_feature_map)

  (b, c, h, w) = normalized_feature_map.shape
  normalized_feature_map_hw_collapsed = normalized_feature_map.reshape(b, c, h * w)
  mean = torch.mean(normalized_feature_map_hw_collapsed, dim=1).unsqueeze(dim=1) # b x 1 x h*w
  diffs = torch.linalg.norm(normalized_feature_map_hw_collapsed - mean, dim=-1) # b x c
  var = torch.mean(torch.pow(diffs, 2.0), dim=1).unsqueeze(dim=1) # b x 1
  zscores = (diffs / torch.pow(var, 0.5)).squeeze() # b x c
  channel_kurt = torch.mean(torch.pow(zscores, 4.0), dim=1) # b
  


  # print(zscores.shape, channel_kurt.shape, var.shape, diffs.shape, mean.shape)

  return torch.mean(channel_kurt) 

