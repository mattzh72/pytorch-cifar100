import torch

from utils import cache_intermediate_output

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


def compute_feature_map_covariance_distance_from_identity(feature_map, normalize_by_channels = True):
    """
    feature_map: (b x c x h x w)
    """
    b,c,h,w = feature_map.shape

    
    # assert feature_map_has_0_mean_1_var(feature_map)
    
    
    covariance_matrix = compute_feature_map_covariance(feature_map)
    
    distance = torch.linalg.norm(covariance_matrix - torch.eye(c).cuda())

    #get rid of off_diagonal entries

    if normalize_by_channels:
        distance = distance/(c**2)

    return distance


def get_whitening_conv1x1s(net, get_excluded_layers=False):
    whitening_conv1x1s = {}
    excluded = []
    for name, layer in net.named_modules():
        if 'whitening' in name:
          whitening_conv1x1s[name] = layer
        else:
          excluded.append(layer)

    return whitening_conv1x1s, excluded


def get_whitening_conv1x1_feature_map_cache(net):
    whitening_conv1x1_feature_map_cache = {}
    for name, layer in get_whitening_conv1x1s(net):
        layer.register_forward_hook(cache_intermediate_output(name, whitening_conv1x1_feature_map_cache))

    return whitening_conv1x1_feature_map_cache
