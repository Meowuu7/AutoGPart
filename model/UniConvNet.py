import torch
import torch.nn as nn
from .point_convolution_universal import TransitionDown, TransitionUp
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
from .utils import farthest_point_sampling, get_knn_idx, batched_index_select
from .DGCNN import get_graph_feature_with_normals, get_graph_feature


class PointNetPP_DownModule(nn.Module):
    def __init__(self, mlp, mlp_in, r, n_sample, args=None):
        super(PointNetPP_DownModule, self).__init__()
        self.mlp = mlp
        self.mlp_in = mlp_in
        self.r = r
        self.n_sample = n_sample
        self.args = args

        # only in prediction layers, we have last_act=False & bn=False
        self.mlp_conv_layer = construct_conv_modules(
                mlp_dims=mlp, n_in=mlp_in[0],
                last_act=True,
                bn=True
            )

    def sample_and_group(self, feat, pos, n_samples, use_pos=True, k=64):
        bz, N = pos.size(0), pos.size(1)
        fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=n_samples)
        # bz x n_samples x pos_dim
        sampled_pos = pos.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples, -1)
        ppdist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        ppdist = torch.sqrt(ppdist)
        topk_dist, topk_idx = torch.topk(ppdist, k=k, dim=2, largest=False)
        grouped_pos = batched_index_select(values=pos, indices=topk_idx, dim=1)
        grouped_pos = grouped_pos - sampled_pos.unsqueeze(2)
        if feat is not None:
            grouped_feat = batched_index_select(values=feat, indices=topk_idx, dim=1)
            if use_pos:
                grouped_feat = torch.cat([grouped_pos, grouped_feat], dim=-1)
        else:
            grouped_feat = grouped_pos
        return grouped_feat, topk_dist, sampled_pos

    def max_pooling_with_r(self, grouped_feat, ppdist, r=None):
        if r is None:
            res, _ = torch.max(grouped_feat, dim=2)
        else:
            indicators = (ppdist <= r).float()
            indicators_expand = indicators.unsqueeze(-1).repeat(1, 1, 1, grouped_feat.size(-1))
            indicators[indicators < 0.5] = -1e8
            grouped_feat[indicators_expand < 0.5] = -1e8
            res, _ = torch.max(grouped_feat, dim=2)
        return res

    def forward(self, x, pos):
        bz = pos.size(0)
        if self.n_sample == 1:
            grouped_feat = x.unsqueeze(1)
            grouped_feat = torch.cat(
                [pos.unsqueeze(1), grouped_feat], dim=-1
            )
            grouped_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
                grouped_feat, self.mlp_conv_layer
            ).squeeze(1)
            x, _ = torch.max(grouped_feat, dim=1, keepdim=True)
            sampled_pos = torch.zeros((bz, 1, 3), dtype=torch.float, device=pos.device)
            pos = sampled_pos
        else:
            grouped_feat, topk_dist, pos = self.sample_and_group(x, pos, self.n_sample, use_pos=True, k=64)
            grouped_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
                grouped_feat, self.mlp_conv_layer
            )
            cur_radius = self.r
            # Group features via max-pooling
            x = self.max_pooling_with_r(grouped_feat, topk_dist, r=cur_radius)
        return x, pos

# up-conv module
class PointNetPP_UpModule(nn.Module):
    def __init__(self, mlp, mlp_in, with_prev_feat=True, itp=True, args=None):
        super(PointNetPP_UpModule, self).__init__()
        self.mlp = mlp
        self.mlp_in = mlp_in
        self.args = args
        self.with_prev_feat = with_prev_feat
        self.itp = itp

        # only in prediction layers, we have last_act=False & bn=False
        self.mlp_conv_layer = construct_conv_modules(
            mlp_dims=mlp, n_in=mlp_in,
            last_act=True,
            bn=True
        )

    def interpolate_features(self, feat, p1, p2, ):
        # print(f"Interpolate features, feat.size = {feat.size()}, p1.size = {p1.size()}, p2.size = {p2.size()}")
        dist = p2[:, :, None, :] - p1[:, None, :, :]
        dist = torch.norm(dist, dim=-1, p=2, keepdim=False)
        topkk = min(3, dist.size(-1))
        dist, idx = dist.topk(topkk, dim=-1, largest=False)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # weight.size() = bz x N2 x 3; idx.size() = bz x N2 x 3
        three_nearest_features = batched_index_select(feat, idx, dim=1)  # 1 is the idx dimension
        interpolated_feats = torch.sum(three_nearest_features * weight[:, :, :, None], dim=2, keepdim=False)
        return interpolated_feats

    def forward(self, x, pos, prev_x, prev_pos, last_up_layer=False):
        # GET interpolated features via x, pos and prev_pos
        if self.itp:
            interpolated_feats = self.interpolate_features(x, pos, prev_pos)
        else:
            interpolated_feats = x
        # GET prev_x feature
        if prev_x is None:
            prev_x = prev_pos
        elif last_up_layer:
            prev_x = torch.cat([prev_x, prev_pos], dim=-1)
        # print(interpolated_feats.size(), prev_x.size())
        if self.with_prev_feat:
            cur_up_feats = torch.cat([interpolated_feats, prev_x], dim=-1)
        else:
            cur_up_feats = interpolated_feats
        x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            cur_up_feats.unsqueeze(2), self.mlp_conv_layer
        ).squeeze(2)
        pos = prev_pos
        return x, pos

class DGCNN_ECModule(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, nn_nb=80):
        super(DGCNN_ECModule, self).__init__()
        # 1. get graph feature from input features; 2. get per-point feature after convolution from edge feature via
        # convolution modules; 3. max pooling for per-point feature
        self.k = nn_nb
        self.bn = nn.GroupNorm(2, out_feat_dim)
        self.EC_conv_layers = nn.Sequential(nn.Conv2d(in_feat_dim * 2, out_feat_dim, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x,):
        # x.size = bz x N x feat_dim
        x = x.contiguous().transpose(-1, -2).contiguous()
        if x.size(1) > 3:
            x = get_graph_feature_with_normals(x, k1=self.k, k2=self.k)
        else:
            x = get_graph_feature(x, k1=self.k, k2=self.k)
        x = self.EC_conv_layers(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = x.contiguous().transpose(-1, -2).contiguous()
        return x


# a feature mapping layer + several convolution modules
class UniConvNet(nn.Module):
    def __init__(self, in_feat_dim: int, args=None):
        '''
        :param in_feat_dim:
        :param args:
        '''
        super(UniConvNet, self).__init__()

        self.skip_global = False

        # in_feat_dim = 6, first-layer dim = 128,
        self.n_samples = [512, 128, 1] if "motion" not in args.task else [256, 128, 1]
        if args.test_performance or ("motion" in args.task):
            mlps = [[64,64,128], [128,128,256], [256,512,1024]]
            mlps_in = [[in_feat_dim,64,64], [128+3,128,128], [256+3,256,512]]

            up_mlps = [[256, 256], [256, 128], [128, 128, 128]]
            up_mlps_in = [1024 + 256, 256 + 128, 128 + in_feat_dim]
            up_mlps_in_prev_feat_sz = [256, 128, in_feat_dim]
        else:
            mlps = [[32, 32, 64], [64, 64, 128], [128, 256, 256]]
            mlps_in = [[in_feat_dim, 32, 32], [64 + 3, 64, 64], [128 + 3, 128, 256]]

            up_mlps = [[128, 128], [128, 64], [64, 64, 64]]
            up_mlps_in = [256 + 128, 128 + 64, 64 + in_feat_dim]
            up_mlps_in_prev_feat_sz = [128, 64, in_feat_dim]

        self.in_feat_dim = in_feat_dim
        self.radius = [0.2, 0.4, None]

        # mlps, mlps_in
        self.pnpp_down_conv_layers = nn.ModuleList()
        for mlp_in, mlp, n_sample, r in zip(mlps_in, mlps, self.n_samples, self.radius):
            cur_down_conv_layer = PointNetPP_DownModule(mlp, mlp_in, r, n_sample, args=args)
            self.pnpp_down_conv_layers.append(cur_down_conv_layer)

        # mlp, mlp_in, with_prev_feat=True, args=None
        self.pnpp_up_conv_layers = nn.ModuleList()
        for mlp_in, mlp in zip(up_mlps_in, up_mlps):
            cur_up_conv_layer = PointNetPP_UpModule(mlp, mlp_in, with_prev_feat=True, args=args)
            self.pnpp_up_conv_layers.append(cur_up_conv_layer)

        # mlp, mlp_in, with_prev_feat=True, args=None
        self.pnpp_up_conv_layers_no_itp = nn.ModuleList()
        for mlp_in, mlp, mlp_in_prev_feat in zip(up_mlps_in, up_mlps, up_mlps_in_prev_feat_sz):
            cur_up_conv_layer = PointNetPP_UpModule(mlp, mlp_in - mlp_in_prev_feat, with_prev_feat=False, args=args)
            self.pnpp_up_conv_layers_no_itp.append(cur_up_conv_layer)

        self.dgcnn_ec_conv_layers = nn.ModuleList()
        for i, (mlp_in, mlp) in enumerate(zip(mlps_in, mlps)):
            if i == 0:
                cur_dgcnn_ec_conv_layer = DGCNN_ECModule(mlp_in[0], mlp[-1])
            else:
                cur_dgcnn_ec_conv_layer = DGCNN_ECModule(mlp_in[0] - 3, mlp[-1])
            self.dgcnn_ec_conv_layers.append(cur_dgcnn_ec_conv_layer)

        self.dgcnn_ec_up_conv_layers = nn.ModuleList()
        for mlp_in, mlp in zip(up_mlps_in, up_mlps):
            cur_dgcnn_ec_up_conv_layer = PointNetPP_UpModule(mlp, mlp_in, with_prev_feat=True, itp=False, args=args)
            self.dgcnn_ec_up_conv_layers.append(cur_dgcnn_ec_up_conv_layer)

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, return_global=False,
                conv_select_types=[0, 0, 0, 0, 0],):
        # conv_layer_type: 0 -- PointNetPP module, 1 -- EC, 2 -- PointNetPP no-up module, 3 -- skip

        bz = pos.size(0)

        cache = []
        cache.append((None if x is None else x.clone(), pos.clone()))

        real_selected_types = [jj for jj in conv_select_types if jj != 3]

        down_conv_modules = []
        up_conv_modules = []

        for i, i_conv_ty in enumerate(real_selected_types):
            if i_conv_ty == 0:
                cur_down_conv_module = self.pnpp_down_conv_layers[i]
                cur_up_conv_module = self.pnpp_up_conv_layers[-i-1]
            elif i_conv_ty == 1:
                cur_down_conv_module = self.dgcnn_ec_conv_layers[i]
                cur_up_conv_module = self.dgcnn_ec_up_conv_layers[-i-1]
            elif i_conv_ty == 2:
                cur_down_conv_module = self.pnpp_down_conv_layers[i]
                cur_up_conv_module = self.pnpp_up_conv_layers_no_itp[-i-1]
            else:
                raise ValueError(f"Unrecognized convolution module type: {i_conv_ty}.")
            down_conv_modules.append(cur_down_conv_module)
            up_conv_modules = [cur_up_conv_module] + up_conv_modules

        # down_conv_modules
        for i, (n_sample, down_conv_module) in enumerate(zip(self.n_samples, down_conv_modules)):
            if real_selected_types[i] in [0,2]:
                x, pos = down_conv_module(x, pos)
                # print(x.size(), pos.size())
            elif real_selected_types[i] in [1]:
                if x is not None and i == 0:
                    x = torch.cat([pos, x], dim=-1)
                    # cache[0] = (x.clone(), pos.clone())
                if x is None:
                    x = pos.clone()
                    # cache[0] = (x.clone(), pos.clone())
                x = down_conv_module(x)
            else:
                raise ValueError(f"Unrecognized convolution module type: {real_selected_types[i]}.")
            cache.append((x.clone(), pos.clone()))
        # remember global level feature
        global_x = x.clone()
        # x, pos, prev_x, prev_pos, last_up_layer=False
        for i, up_conv_module in enumerate(up_conv_modules):
            prev_x, prev_pos = cache[-i - 2][0], cache[-i - 2][1]
            # print(prev_x.size(), prev_pos.size(), x.size(), pos.size())
            x, pos = up_conv_module(x, pos, prev_x, prev_pos, last_up_layer=(i == len(up_conv_modules) - 1))

        if return_global:
            return x, global_x, pos
        else:
            return x, pos
