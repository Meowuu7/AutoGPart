import torch
try:
    from torch_cluster import fps
except:
    pass

import torch.nn as nn
import numpy as np

try:
    import open3d as o3d
except:
    pass

def set_bn_not_training(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_bn_not_training(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            if isinstance(block, nn.BatchNorm1d) or isinstance(block, nn.BatchNorm2d):
                block.is_training = False
    else:
        raise ValueError("Not recognized module to set not training!")

def set_grad_to_none(module):
    if isinstance(module, nn.ModuleList):
        for block in module:
            set_grad_to_none(block)
    elif isinstance(module, nn.Sequential):
        for block in module:
            for param in block.parameters():
                param.grad = None
    else:
        raise ValueError("Not recognized module to set not training!")


def init_weight(blocks):
    for module in blocks:
        if isinstance(module, nn.Sequential):
            for subm in module:
                if isinstance(subm, nn.Linear):
                    nn.init.xavier_uniform_(subm.weight)
                    nn.init.zeros_(subm.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def construct_conv1d_modules(mlp_dims, n_in, last_act=True, bn=True, others_bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        if i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act):
            # if others_bn and ouc % 4 == 0:
            if others_bn: # and ouc % 4 == 0:
                blk = nn.Sequential(
                        nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                        nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                    # nn.GroupNorm(num_groups=4, num_channels=ouc),
                        nn.ReLU()
                    )
            else:
                blk = nn.Sequential(
                    nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
                nn.BatchNorm1d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv1d(in_channels=inc, out_channels=ouc, kernel_size=(1,), stride=(1,), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


def construct_conv_modules(mlp_dims, n_in, last_act=True, bn=True):
    rt_module_list = nn.ModuleList()
    for i, dim in enumerate(mlp_dims):
        inc, ouc = n_in if i == 0 else mlp_dims[i-1], dim
        # if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act))  and ouc % 4 == 0:
        if (i < len(mlp_dims) - 1 or (i == len(mlp_dims) - 1 and last_act)): #  and ouc % 4 == 0:
            blk = nn.Sequential(
                    nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                    nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
                    nn.ReLU()
                )
        # elif bn  and ouc % 4 == 0:
        elif bn: #  and ouc % 4 == 0:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
                nn.BatchNorm2d(num_features=ouc, eps=1e-5, momentum=0.1),
                # nn.GroupNorm(num_groups=4, num_channels=ouc),
            )
        else:
            blk = nn.Sequential(
                nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=(1, 1), stride=(1, 1), bias=True),
            )
        rt_module_list.append(blk)
    init_weight(rt_module_list)
    return rt_module_list


class CorrFlowPredNet(nn.Module):
    def __init__(self, corr_feat_dim: int=32):
        super(CorrFlowPredNet, self).__init__()

    @staticmethod
    def apply_module_with_conv2d_bn(x, module):
        x = x.transpose(2, 3).contiguous().transpose(1, 2).contiguous()
        # print(x.size())
        for layer in module:
            for sublayer in layer:
                x = sublayer(x.contiguous())
            x = x.float()
        x = torch.transpose(x, 1, 2).transpose(2, 3)
        return x

    @staticmethod
    def apply_module_with_conv1d_bn(x, module):
        x = x.transpose(1, 2).contiguous()
        # print(x.size())
        for layer in module:
            for sublayer in layer:
                x = sublayer(x.contiguous())
            x = x.float()
        x = torch.transpose(x, 1, 2)
        return x

def estimate_normals(pos):
    # pos.size = bz x N x 3
    normals = []
    for i in range(pos.size(0)):
        pts = pos[i].detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        nms = np.array(pcd.normals)
        normals.append(torch.from_numpy(nms).to(pos.device).float().unsqueeze(0))
    normals = torch.cat(normals, dim=0)
    return normals
