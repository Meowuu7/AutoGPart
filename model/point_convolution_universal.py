import torch
# from torch_cluster import fps
import torch.nn as nn
from .utils import farthest_point_sampling, get_knn_idx, batched_index_select

from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet


def max_pooling_with_r(grouped_feat, ppdist, r=None):
    if r is None:
        res, _ = torch.max(grouped_feat, dim=2)
    else:
        # print(f"max pooling with r = {r}")
        # bz x N x k
        indicators = (ppdist > r).float()
        # indicators_expand = indicators.unsqueeze(-1).repeat(1, 1, 1, grouped_feat.size(-1))
        indicators[indicators > 0.5] = -1e8
        # print(grouped_feat.size(), indicators.size())
        grouped_feat = grouped_feat + indicators.unsqueeze(-1)

        # grouped_feat[indicators_expand < 0.5] = -1e8
        res, _ = torch.max(grouped_feat, dim=2)
    return res


class PointppConv(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, k=16, last_act=False):
        super(PointppConv, self).__init__()

        self.k, self.in_feat_dim, self.out_feat_dim = k, in_feat_dim, out_feat_dim

        self.last_act = last_act

        # ###### define feature transformation net
        self.feat_mlp_net = construct_conv_modules(
            [out_feat_dim, out_feat_dim], n_in=in_feat_dim, last_act=last_act, bn=True
        )

        #### Do not need to map gathered features again, actually ####
        # self.feat_mlp_glb_out_net = construct_conv_modules(
        #     [out_feat_dim, out_feat_dim], n_in=out_feat_dim, last_act=last_act, bn=True
        # )

    def forward(self, x, pos, sampled_idx=None, relative_pos=False, n_sampling=1, r=None):
        # x.size = bz x N x feat_dim
        bz, N = pos.size(0), pos.size(1)

        if n_sampling == 1:
            sampled_pos = torch.zeros((bz, 1, 3), dtype=torch.float32, device=pos.device)
            gathered_pos = pos.unsqueeze(1)
        else:
            if sampled_idx is None:
                # sampled_idx = torch.arange(N).unsqueeze(0).repeat(bz, 1).to(x.device)
                sampled_idx = torch.arange(N * bz, device=x.device)
            # # calculate distance between sampled points and other points
            # print(pos.size(), sampled_idx.size(), x.size())
            # sampled_pos = batched_index_select(values=pos, indices=sampled_idx, dim=1)
            sampled_pos = pos.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, pos.size(-1))
            # sampled_pos.size = bz x nsmp x 3
            sampled_dist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
            # sampled_dist.size = bz x nsmp x N
            nearest_k_dist, nearest_k_idx = torch.topk(sampled_dist, k=self.k, largest=False)
            # # bz x nsmp x k
            # # !!!! we assume that `pos` is not included in `x` !!!!
            gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)

        if x is None:
            gathered_pos = gathered_pos - sampled_pos.unsqueeze(2)
            gathered_feat = gathered_pos
        else:
            # print(x.size(), nearest_k_idx.size(), )
            if len(x.size()) > 3:
                x = x.squeeze(-2)
            if n_sampling == 1:
                gathered_x = x.unsqueeze(1)
            else:
                gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
            # if relative_pos:
            gathered_pos = gathered_pos - sampled_pos.unsqueeze(2)
            # print(gathered_x.size(), gathered_pos.size())
            # print(x.size(), pos.size(), gathered_x.size(), gathered_pos.size())
            gathered_feat = torch.cat(
                [gathered_x, gathered_pos], dim=-1
            )
        # # bz x N x k x feat_dim
        gathered_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
            gathered_feat, self.feat_mlp_net
        )

        if n_sampling == 1:
            gathered_feat, _ = torch.max(gathered_feat, dim=2, keepdim=False)
        else:
            gathered_feat = max_pooling_with_r(gathered_feat, nearest_k_dist, r=r)

        # gathered_feat, _ = torch.max(gathered_feat, dim=2, keepdim=False)
        # gathered_feat = max_pooling_with_r(nearest_k_x=gathered_feat, nearest_k_idx=nearest_k_idx, r=r, pos=x)
        # gathered_feat = torc

        return gathered_feat


class EdgeConvConv(nn.Module):
    def __init__(self, node_feat_in_dim, node_feat_out_dim, k=16, last_act=False):
        super(EdgeConvConv, self).__init__()
        # # in_dim, out_dim, k
        self.k = k
        self.node_feat_in_dim = node_feat_in_dim
        self.node_feat_out_dim = node_feat_out_dim
        self.mlp_in_dim = node_feat_in_dim * 2
        self.mlp_out_dim = node_feat_out_dim
        self.conv_mlp = construct_conv_modules(
            [self.mlp_in_dim // 2, self.mlp_out_dim], n_in=self.mlp_in_dim, last_act=last_act
        )

    def build_similarity_distance_graph(self, x):
        # # x.size = bz x N x feat_dim
        sim_dist = torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=-1)
        nearest_k_dist, nearest_k_idx = torch.topk(sim_dist, k=self.k, dim=-1, largest=False)
        # gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
        # # gathered_x.size = bz x N x k x feat_dim
        return nearest_k_idx, nearest_k_dist
        # return nearest_k_idx, gathered_x

    # ###### !!!! set relative_pos to `False` in the following layers...
    def forward(self, x, pos, r=None, relative_pos=False, similarity_calculation_feat=None):
        # if similarity_calculation_feat is None:
        #     similarity_calculation_feat = x
        # # # still use [global transformation, position information] as similarity indicator features
        # ### !!! similarity calculation features ---- the similarity calculation process will not be influenced by
        # `relative_pos` item
        if x is None:
            similarity_calculation_feat = pos
        else:
            similarity_calculation_feat = torch.cat([x, pos], dim=-1)
        nearest_k_idx, nearest_k_dist = self.build_similarity_distance_graph(similarity_calculation_feat)

        gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)
        if x is None:
            # gathered_x = gathered_pos - pos.unsqueeze(2)
            if relative_pos:
                gathered_pos = gathered_pos - pos.unsqueeze(2)
                edge_x = torch.cat(
                    [torch.zeros_like(gathered_pos), gathered_pos], dim=-1
                )
            else:
                edge_x = torch.cat(
                    [pos.unsqueeze(2).repeat(1, 1, self.k, 1), gathered_pos], dim=-1
                )
        else:
            gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
            if relative_pos:
                gathered_pos = gathered_pos - pos.unsqueeze(2) # bz x N x k x 3
                # print(f"Using relative position in EdgeConv, gathered_pos.size = {gathered_pos.size()}")
                edge_x = torch.cat(
                    [x.unsqueeze(2).repeat(1, 1, self.k, 1), torch.zeros_like(gathered_pos),
                     gathered_x, gathered_pos], dim=-1
                )
            else:
                edge_x = torch.cat(
                    [x.unsqueeze(2).repeat(1, 1, self.k, 1), pos.unsqueeze(2).repeat(1, 1, self.k, 1),
                     gathered_x, gathered_pos], dim=-1
                )

        edge_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            edge_x, self.conv_mlp
        )
        node_x = max_pooling_with_r(nearest_k_x=edge_x, nearest_k_idx=nearest_k_idx, r=r, pos=x)

        return node_x


class PointwebConv(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, k=16, last_act=False):
        super(PointwebConv, self).__init__()

        self.k, self.in_feat_dim, self.out_feat_dim = k, in_feat_dim, out_feat_dim

        self.last_act = last_act

        # ###### define feature transformation net
        self.feat_mlp_net = construct_conv_modules(
            [out_feat_dim, out_feat_dim], n_in=in_feat_dim, last_act=last_act, bn=True
        )

        self.feat_diff_to_weights_net = construct_conv_modules(
            [out_feat_dim, out_feat_dim], n_in=in_feat_dim, last_act=False,
        )

    def forward(self, x, pos, sampled_idx=None, relative_pos=False):
        # x.size = bz x N x feat_dim
        bz, N, feat_dim = x.size(0), x.size(1), x.size(2)
        if sampled_idx is None:
            # sampled_idx = torch.arange(N).unsqueeze(0).repeat(bz, 1).to(x.device)
            sampled_idx = torch.arange(N * bz, device=x.device)
        # # calculate distance between sampled points and other points
        # print(pos.size(), sampled_idx.size(), x.size())
        # sampled_pos = batched_index_select(values=pos, indices=sampled_idx, dim=1)
        sampled_pos = pos.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, pos.size(-1))
        # # sampled_pos.size = bz x nsmp x 3
        sampled_dist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        # # sampled_dist.size = bz x nsmp x N
        nearest_k_dist, nearest_k_idx = torch.topk(sampled_dist, k=self.k, largest=False)
        # # bz x nsmp x k
        # # !!!! we assume that `pos` is not included in `x` !!!!
        gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)
        gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
        # if relative_pos:
        relative_gathered_pos = gathered_pos - sampled_pos.unsqueeze(2)
        sampled_x = x.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, x.size(-1))
        relative_gathered_x = gathered_x - sampled_x.unsqueeze(2)

        # # if relative_pos:
        # gathered_feat = torch.cat(
        #     [gathered_x, relative_gathered_pos if relative_pos else gathered_pos], dim=-1
        # )

        relative_gathered_feat = torch.cat(
            [relative_gathered_x, relative_gathered_pos], dim=-1
        )
        # # feat_diff

        weights = CorrFlowPredNet.apply_module_with_conv2d_bn(
            relative_gathered_feat, self.feat_diff_to_weights_net
        )
        weights = torch.sigmoid(weights)

        sampled_x_expand = sampled_x.unsqueeze(2)
        weights_self = CorrFlowPredNet.apply_module_with_conv2d_bn(
            sampled_x_expand, self.feat_diff_to_weights_net
        )
        weights_self = torch.sigmoid(weights_self)

        res_x = sampled_x_expand + weights_self * sampled_x_expand - torch.sum(relative_gathered_feat * weights, dim=2, keepdim=True)

        gathered_feat = res_x

        # # bz x N x k x feat_dim
        gathered_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
            gathered_feat, self.feat_mlp_net
        )

        gathered_feat, _ = torch.max(gathered_feat, dim=2, keepdim=False)
        # gathered_feat = max_pooling_with_r(nearest_k_x=gathered_feat, nearest_k_idx=nearest_k_idx, r=r, pos=x)

        return gathered_feat


class PointwiseConv(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, k=16, last_act=False):
        super(PointwiseConv, self).__init__()

        self.k, self.in_feat_dim, self.out_feat_dim = k, in_feat_dim, out_feat_dim

        self.last_act = last_act

        # ###### define feature transformation net
        self.feat_mlp_net = construct_conv_modules(
            [out_feat_dim, out_feat_dim], n_in=in_feat_dim, last_act=last_act, bn=True
        )

    def forward(self, x, pos, sampled_idx=None, relative_pos=False):
        # x.size = bz x N x feat_dim
        bz, N, feat_dim = x.size(0), x.size(1), x.size(2)
        if sampled_idx is None:
            # sampled_idx = torch.arange(N).unsqueeze(0).repeat(bz, 1).to(x.device)
            sampled_idx = torch.arange(N * bz, device=x.device)
        # # calculate distance between sampled points and other points
        # print(pos.size(), sampled_idx.size(), x.size())
        # sampled_pos = batched_index_select(values=pos, indices=sampled_idx, dim=1)
        sampled_pos = pos.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, pos.size(-1))
        sampled_x = x.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, x.size(-1))
        # # sampled_pos.size = bz x nsmp x 3
        sampled_dist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        dist = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1) # sampled distanc

        # # sampled_dist.size = bz x nsmp x N
        nearest_k_dist, nearest_k_idx = torch.topk(sampled_dist, k=self.k, largest=False)
        # # bz x nsmp x k
        # # !!!! we assume that `pos` is not included in `x` !!!!
        gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)
        gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
        if relative_pos:
            gathered_pos = gathered_pos - sampled_pos.unsqueeze(2)
        gathered_feat = torch.cat(
            [gathered_x, gathered_pos], dim=-1
        )
        # relative_x_feat = torch.cat(
        #     [sampled_x, torch.zeros((bz, sampled_x.size(1), pos.size(-1)), device=pos.device, dtype=torch.float32)],
        #     dim=-1
        # )

        avg_gathered_feat = torch.mean(gathered_feat, dim=2, keepdim=True)

        # # bz x N x k x feat_dim
        gathered_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
            avg_gathered_feat, self.feat_mlp_net
        )

        gathered_feat = gathered_feat.squeeze(2)
        # gathered_feat, _ = torch.max(gathered_feat, dim=2, keepdim=False)
        # gathered_feat = max_pooling_with_r(nearest_k_x=gathered_feat, nearest_k_idx=nearest_k_idx, r=r, pos=x)

        return gathered_feat


class RelationConv(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, k=16, last_act=False):
        super(RelationConv, self).__init__()

        self.k, self.in_feat_dim, self.out_feat_dim = k, in_feat_dim, out_feat_dim

        self.last_act = last_act

        self.feat_mlp_net = construct_conv_modules(
            [out_feat_dim, out_feat_dim], n_in=in_feat_dim, last_act=last_act, bn=True
        )

        self.relation_vec_to_weights_net = construct_conv_modules(
            [32, in_feat_dim, in_feat_dim], n_in=10, last_act=False, bn=False
        )

    def forward(self, x, pos, sampled_idx=None, relative_pos=False):
        # x.size = bz x N x feat_dim
        bz, N, feat_dim = x.size(0), x.size(1), x.size(2)
        if sampled_idx is None:
            # sampled_idx = torch.arange(N).unsqueeze(0).repeat(bz, 1).to(x.device)
            sampled_idx = torch.arange(N * bz, device=x.device)
        # # calculate distance between sampled points and other points
        # print(pos.size(), sampled_idx.size(), x.size())
        # sampled_pos = batched_index_select(values=pos, indices=sampled_idx, dim=1)
        sampled_pos = pos.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, pos.size(-1))
        # sampled_x = x.contiguous().view(bz * N, -1)[sampled_idx, :].contiguous().view(bz, -1, x.size(-1))
        # # sampled_pos.size = bz x nsmp x 3
        sampled_dist = torch.sum((sampled_pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        # dist = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1) # sampled distanc

        # # sampled_dist.size = bz x nsmp x N
        nearest_k_dist, nearest_k_idx = torch.topk(sampled_dist, k=self.k, largest=False)
        # # bz x nsmp x k
        # # !!!! we assume that `pos` is not included in `x` !!!!
        gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)
        gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)

        relative_pos_vec = gathered_pos - sampled_pos.unsqueeze(2) # relative position
        euclidean_dis = torch.norm(relative_pos_vec, p=2, dim=-1, keepdim=True) # point-point distance
        # relation vector = [center point absolute position, neighbour point absolute position, relative position, dis]
        relation_vec = torch.cat(
            [sampled_pos.unsqueeze(2).repeat(1, 1, self.k, 1), gathered_pos, relative_pos_vec, euclidean_dis], dim=-1
        )

        # conv_weights
        conv_weights = CorrFlowPredNet.apply_module_with_conv2d_bn(
            relation_vec, self.relation_vec_to_weights_net
        )
        #
        if relative_pos:
            gathered_pos = gathered_pos - sampled_pos.unsqueeze(2)
        gathered_feat = torch.cat(
            [gathered_x, gathered_pos], dim=-1
        )

        # conv...
        gathered_feat = conv_weights * gathered_feat
        gathered_feat, _ = torch.max(gathered_feat, dim=2, keepdim=True) # max pooling
        gathered_feat = torch.relu(gathered_feat)

        # # bz x N x k x feat_dim

        gathered_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
            gathered_feat, self.feat_mlp_net
        )

        gathered_feat = gathered_feat.squeeze(2) # squeeze 1-dim
        # gathered_feat = max_pooling_with_r(nearest_k_x=gathered_feat, nearest_k_idx=nearest_k_idx, r=r, pos=x)

        return gathered_feat

class LocalPNConvNet(nn.Module):
    def __init__(self, node_feat_dim, mlp_dims, k=16):
        super(LocalPNConvNet, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.k = k
        self.cn_feat_dim = mlp_dims[-1]
        # # already after init
        self.local_feat_transformation = construct_conv_modules(
            mlp_dims, n_in=node_feat_dim, last_act=False
        )

        self.glb_feat_transformation = construct_conv_modules(
            [mlp_dims[-1] * 2, mlp_dims[-1]], n_in=mlp_dims[-1], last_act=False
        )

        # #
        self.combined_feat_transformation = construct_conv_modules(
            [self.cn_feat_dim, self.cn_feat_dim], n_in=self.cn_feat_dim * 2, last_act=False
        )

        self.combined_glb_feat_transformation = construct_conv_modules(
            [self.cn_feat_dim * 2, self.cn_feat_dim], n_in=self.cn_feat_dim, last_act=False
        )

    def forward(self, x, pos, relative_pos=False):
        dists = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        nearest_k_dist, nearest_k_idx = torch.topk(dists, k=self.k, dim=-1, largest=False)

        gathered_pos = batched_index_select(values=pos, indices=nearest_k_idx, dim=1)

        if relative_pos:
            gathered_pos = gathered_pos - pos.unsqueeze(2)

        if x is not None:
            gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)

            gathered_x = torch.cat(
                [gathered_x, gathered_pos], dim=-1
            )
        else:
            gathered_x = gathered_pos
        # # bz x N x k x feat_dim

        # if relative_pos and gathered_x.size(-1) == 6:
            # # then transform the input position feature
            # gathered_x[:, :, :, :3] = gathered_x[:, :, :, :3] - x.unsqueeze(2)[:, :, :, :3]

        gathered_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            gathered_x, self.local_feat_transformation
        )

        glb_x, _ = torch.max(gathered_x, dim=2, keepdim=True)
        glb_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            glb_x, self.glb_feat_transformation
        )

        combined_x = torch.cat(
            [gathered_x, glb_x.repeat(1, 1, self.k, 1)], dim=-1
        )

        combined_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            combined_x, self.combined_feat_transformation
        )

        combined_glb_x, _ = torch.max(
            combined_x, dim=2, keepdim=True
        )

        combined_glb_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            combined_glb_x, self.combined_glb_feat_transformation
        )

        # # bz x N x 1 x feat_dim
        combined_glb_x = combined_glb_x.squeeze(2)

        return combined_glb_x


class TransitionDown(nn.Module):
    def __init__(self, feat_dim: int, out_feat_dim: int, k: int=16, last_act=True,
                 conv_type="pointnetpp", abstract=True):
        super().__init__()

        self.k = k
        self.out_feat_dim = out_feat_dim
        self.last_act = last_act
        self.abstract = abstract

        self.conv_type = conv_type
        if self.conv_type == "pointnetpp":
            self.feat_conv_net = PointppConv(
                in_feat_dim=feat_dim, out_feat_dim=out_feat_dim, k=self.k, last_act=last_act, )
        elif self.conv_type == "edgeconv":
            self.feat_conv_net = EdgeConvConv(
                node_feat_in_dim=feat_dim, node_feat_out_dim=out_feat_dim, k=self.k, last_act=last_act
            )

        elif self.conv_type == "localconv":
            # node_feat_dim, mlp_dims
            self.feat_conv_net = LocalPNConvNet(
                node_feat_dim=feat_dim, mlp_dims=[out_feat_dim, out_feat_dim], k=self.k
            )
        elif self.conv_type == "rsconv":
            self.feat_conv_net = RelationConv(
                in_feat_dim=feat_dim, out_feat_dim=out_feat_dim, k=self.k, last_act=last_act
            )
        else:
            raise ValueError(f"Not recognized conv_type: {self.conv_type}.")

    def forward(self, x: torch.FloatTensor, pos: torch.FloatTensor, n_sampling: int=None, relative_pos=False,
                return_fps=False, r=None, similarity_calculation_feat=None):
        bz, N = pos.size(0), pos.size(1)
        # fps sampling for set abstract
        if self.abstract:
            assert n_sampling is not None
            fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=n_sampling)
        else:
            fps_idx = None

        # ###### !!!! We assume `pos` is not included in `x` !!!!

        if self.conv_type == "pointnetpp":
            x = self.feat_conv_net(x=x, pos=pos, sampled_idx=fps_idx, relative_pos=relative_pos, n_sampling=n_sampling, r=r)
        elif self.conv_type == "edgeconv":
            # ###### !!!! If relative_pos is True, assume the first three dimensions are `pos` information
            x = self.feat_conv_net(
                x=x, pos=pos, r=r, relative_pos=relative_pos, similarity_calculation_feat=similarity_calculation_feat)
            if self.abstract:
                x = batched_index_select(values=x.contiguous().view(bz * N, -1), indices=fps_idx, dim=0)
                x = x.contiguous().view(bz, n_sampling, -1)
        elif self.conv_type == "localconv":
            x = self.feat_conv_net(x=x, pos=pos, relative_pos=True)
            if self.abstract:
                x = batched_index_select(values=x.contiguous().view(bz * N, -1), indices=fps_idx, dim=0)
                x = x.contiguous().view(bz, n_sampling, -1)
        elif self.conv_type == "rsconv":
            x = self.feat_conv_net(
                x=x, pos=pos, sampled_idx=fps_idx, relative_pos=relative_pos
            )
        else:
            raise ValueError("Not recognized conv_type!")

        # bz x n_sampling x feat_dim

        if self.abstract:
            if n_sampling == 1:
                sampled_pos = torch.zeros((bz, 1, 3), dtype=torch.float32, device=pos.device)
            else:
                sampled_pos = batched_index_select(values=pos.view(bz * N, -1), indices=fps_idx, dim=0)
                sampled_pos = sampled_pos.view(bz, n_sampling, -1)
        else:
            sampled_pos = pos

        if return_fps:
            return x, sampled_pos, fps_idx
        else:
            return x, sampled_pos


class TransitionUp(nn.Module):
    def __init__(self, fea_in: int, fea_out: int):
        super(TransitionUp, self).__init__()

        self.mlp_in = construct_conv1d_modules(
            [fea_out, fea_out], n_in=fea_in, last_act=False
        )

        self.mlp_out = construct_conv1d_modules(
            [fea_out, fea_out], n_in=fea_out, last_act=False
        )

    def forward(self, x1, p1, x2, p2, interpolate=True, skip_connection=True):
        # x1 = CorrFlowPredNet.apply_module_with_conv1d_bn(
        #     x1, self.mlp_in
        # )
        if interpolate:
            # how to set p1 and p2 is important
            # # Distance calculation
            dist = p2[:, :, None, :] - p1[:, None, :, :]
            dist = torch.norm(dist, dim=-1, p=2, keepdim=False)
            # dist.size() = bz x N2 x N1
            # print(dist.size())
            topkk = min(3, dist.size(-1))
            dist, idx = dist.topk(topkk, dim=-1, largest=False)

            # bz x N2 x 3
            # print(dist.size(), idx.size())
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            # weight.size() = bz x N2 x 3; idx.size() = bz x N2 x 3
            three_nearest_features = batched_index_select(x1, idx, dim=1) # 1 is the idx dimension
            interpolated_feats = torch.sum(three_nearest_features * weight[:, :, :, None], dim=2, keepdim=False)
        else:
            interpolated_feats = x1

        y = torch.cat(
            [interpolated_feats, x2], dim=-1
        )
        y = CorrFlowPredNet.apply_module_with_conv1d_bn(
            y, self.mlp_in
        )

        return y.contiguous(), p2


class EdgeConv(nn.Module):
    def __init__(self, node_feat_in_dim, node_feat_out_dim, k=16):
        super(EdgeConv, self).__init__()
        # # in_dim, out_dim, k
        self.k = k
        self.node_feat_in_dim = node_feat_in_dim
        self.node_feat_out_dim = node_feat_out_dim
        self.mlp_in_dim = node_feat_in_dim * 2
        self.mlp_out_dim = node_feat_out_dim
        self.conv_mlp = construct_conv_modules(
            [self.mlp_in_dim // 2, self.mlp_out_dim], n_in=self.mlp_in_dim, last_act=False
        )

    def build_similarity_distance_graph(self, x):
        # # x.size = bz x N x feat_dim
        sim_dist = torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=-1)
        nearest_k_dist, nearest_k_idx = torch.topk(sim_dist, k=self.k, dim=-1, largest=False)
        gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)
        # # gathered_x.size = bz x N x k x feat_dim
        return nearest_k_idx, gathered_x

    def forward(self, x, r=None, relative_pos=False):
        nearest_k_idx, gathered_x = self.build_similarity_distance_graph(x)
        if relative_pos and x.size(-1) == 6:
            # gathered_x.size = bz x N x k x 6; x.size = bz x N x 6
            gathered_x[:, :, :, :3] = gathered_x[:, :, :, :3] - x.unsqueeze(2)[:, :, :, :3]
            # from absolute position to relative position

        edge_x = torch.cat(
            [x.unsqueeze(2).repeat(1, 1, self.k, 1), gathered_x], dim=-1
        )
        # # bz x N x k x (2 * feat_dim)
        edge_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            edge_x, self.conv_mlp
        )
        # # bz x N x k x node_feat_out_dim
        # node_x, _ = torch.max(edge_x, dim=2)
        # # x -- pos
        node_x = max_pooling_with_r(nearest_k_x=edge_x, nearest_k_idx=nearest_k_idx, r=r, pos=x)

        return node_x


class LocalConvNet(nn.Module):
    def __init__(self, node_feat_dim, mlp_dims, k=16):
        super(LocalConvNet, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.k = k
        self.cn_feat_dim = mlp_dims[-1]
        # # already after init
        self.local_feat_transformation = construct_conv_modules(
            mlp_dims, n_in=node_feat_dim, last_act=False
        )

        self.glb_feat_transformation = construct_conv_modules(
            [mlp_dims[-1] * 2, mlp_dims[-1]], n_in=mlp_dims[-1], last_act=False
        )

        # #
        self.combined_feat_transformation = construct_conv_modules(
            [self.cn_feat_dim, self.cn_feat_dim], n_in=self.cn_feat_dim * 2, last_act=False
        )

        self.combined_glb_feat_transformation = construct_conv_modules(
            [self.cn_feat_dim * 2, self.cn_feat_dim], n_in=self.cn_feat_dim, last_act=False
        )

    def forward(self, x, pos, relative_pos=False):

        dists = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        nearest_k_dist, nearest_k_idx = torch.topk(dists, k=self.k, dim=-1, largest=False)
        gathered_x = batched_index_select(values=x, indices=nearest_k_idx, dim=1)

        # # bz x N x k x feat_dim

        if relative_pos:
            # # then transform the input position feature
            gathered_x[:, :, :, :3] = gathered_x[:, :, :, :3] - x.unsqueeze(2)[:, :, :, :3]

        gathered_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            gathered_x, self.local_feat_transformation
        )

        glb_x, _ = torch.max(gathered_x, dim=2, keepdim=True)
        glb_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            glb_x, self.glb_feat_transformation
        )

        combined_x = torch.cat(
            [gathered_x, glb_x.repeat(1, 1, self.k, 1)], dim=-1
        )

        combined_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            combined_x, self.combined_feat_transformation
        )

        combined_glb_x, _ = torch.max(
            combined_x, dim=2, keepdim=True
        )

        combined_glb_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            combined_glb_x, self.combined_glb_feat_transformation
        )

        # # bz x N x 1 x feat_dim
        combined_glb_x = combined_glb_x.squeeze(2)

        return combined_glb_x