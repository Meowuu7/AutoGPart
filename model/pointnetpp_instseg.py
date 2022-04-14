import torch
import torch.nn as nn
from .pnmem_cell import DynamicRNN
from .utils import farthest_point_sampling, batched_index_select, get_knn_idx
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
import torch.nn.functional as F
from model.utils import iou
from torch.autograd import Variable

from .UNet_universal import UNet as UiUNet
from .point_convolution_universal import LocalConvNet, EdgeConv

import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PointNetPPInstSeg(nn.Module):
    def __init__(self, n_layers: int, feat_dims: list, n_samples: list, map_feat_dim: int,
                 landmark_type: str = "net", use_lm=True, stage=1, raw_feat_dim=1, lm_classes=10,
                 latent_type="clustering", clustering_feat_dim=64, args=None):
        super(PointNetPPInstSeg, self).__init__()
        self.n_layers = n_layers
        self.feat_dims = feat_dims
        self.n_samples = n_samples
        self.feat_dims = self.feat_dims
        self.use_lm = use_lm

        self.cluster_seed_pts = args.cluster_seed_pts

        self.xyz_dim = 3
        # # maximum number of masks that can be processed in the model
        self.mask_dim = args.nmasks
        self.map_feat_dim = map_feat_dim
        # # actually the mark of this model & exp
        self.landmark_type = landmark_type
        # # representation dimension
        self.lm_dim = 64
        self.nsmp = 256
        self.raw_feat_dim = raw_feat_dim
        self.npoints = 512

        self.local_unet_n_layers = 5

        map_feat_dim = 64

        self.conv_uniunet = UiUNet(
            feat_dims=feat_dims[:self.local_unet_n_layers],
            n_samples=n_samples[:self.local_unet_n_layers],
            n_layers=self.local_unet_n_layers,
            in_feat_dim=0, # no extra input features except position information
            map_feat_dim=map_feat_dim,
            need_feat_map=True
        )

        # self.conv_uniunet_further = UiUNet(
        #     feat_dims=feat_dims[:self.local_unet_n_layers],
        #     n_samples=n_samples[:self.local_unet_n_layers],
        #     n_layers=self.local_unet_n_layers,
        #     in_feat_dim=map_feat_dim,  # no extra input features except position information
        #     map_feat_dim=map_feat_dim,
        #     need_feat_map=True
        # )

        self.cls_layers = construct_conv1d_modules(
            [map_feat_dim, map_feat_dim // 2, self.mask_dim], n_in=map_feat_dim, last_act=False
        )

        self.conv_pred_layers = construct_conv1d_modules(
            [512, 256, 128, 64, 32, self.mask_dim], n_in=1024, last_act=False
        )

        # self.local_conv_net_local = LocalConvNet(
        #     node_feat_dim=6,
        #     mlp_dims=[map_feat_dim // 2, map_feat_dim],
        #     k=16
        # )
        #
        # self.local_conv_net_edge_conv = EdgeConv(
        #     node_feat_in_dim=6,
        #     node_feat_out_dim=map_feat_dim,
        #     k=16
        # )

        grp_in_feat = map_feat_dim

        self.lm_feat_to_certain_feat_dim_net = construct_conv1d_modules(
            [map_feat_dim // 2, grp_in_feat], n_in=map_feat_dim, last_act=False
        )

        # ###### Feature transformation for grouping features ######
        self.grp_feat_trans_net = construct_conv_modules(
            [grp_in_feat, grp_in_feat], n_in=grp_in_feat * 2, last_act=True
        ) # 64

        self.grp_feat_trans_net_glb = construct_conv_modules(
            [grp_in_feat, grp_in_feat], n_in=grp_in_feat, last_act=False
        )

        self.grp_feat_combined_net = construct_conv_modules(
            [grp_in_feat, grp_in_feat // 2, 1], n_in=grp_in_feat * 2, last_act=False
        )
        # ###### END Feature transformation for grouping features ######

        self.mode_pred = DynamicRNN(indim=self.cluster_seed_pts, nmasks=args.nmasks)

    # # given nearest k-points indexes and position information, generate local feature discrepancy vector for each point and its adjacent area
    def generate_feature_discrepancy(self, nearest_k_idx, nearest_k_dist, features, fps_idx, r=0.05):
        nearest_k_indicators = (nearest_k_dist <= r).float()
        print(torch.mean(torch.sum(nearest_k_indicators, dim=-1)))
        nearest_k_features = batched_index_select(features, indices=nearest_k_idx, dim=1)
        # bz x N x k x feat_dim
        # bz x N x k
        nearest_k_features_avg = (torch.sum(nearest_k_features * nearest_k_indicators.unsqueeze(-1), dim=2)) / \
                                 (torch.sum(nearest_k_indicators.unsqueeze(-1), dim=2))
        # # need not to consider the situation of zero... since the point at least connects wth itself
        # bz x N x faet_dim
        bz, N, nsmp = features.size(0), features.size(1), nearest_k_idx.size(1)
        features_sub = features.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
        nearest_k_discrepancy = features_sub - nearest_k_features_avg
        return nearest_k_discrepancy

    def generate_feature_sim(self, nearest_k_idx, nearest_k_dist, features, fps_idx, r=0.03):
        bz, N, nsmp = features.size(0), features.size(1), nearest_k_idx.size(1)
        nearest_k_indicators = (nearest_k_dist <= r).float()
        print(torch.mean(torch.sum(nearest_k_indicators, dim=-1)))
        nearest_k_features = batched_index_select(features, indices=nearest_k_idx, dim=1)
        # bz x nsmp x k x feat_dim
        nearest_k_features_split = [nearest_k_features[:, :, :, :3], nearest_k_features[:, :, :, 3:]]
        features_sub = features.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
        features_sub_split = [features_sub[:, :, :3], features_sub[:, :, 3:]]
        # # inner_prod =
        inner_prod_avgs = []
        for nearest_k_feat, feat_sub in zip(nearest_k_features_split, features_sub_split):
            cur_inner_prod = torch.sum(feat_sub.unsqueeze(2) * nearest_k_feat, dim=-1) / \
                             (torch.clamp(torch.norm(feat_sub.unsqueeze(2), dim=-1, p=2) * torch.norm(nearest_k_feat, dim=-1, p=2), min=1e-9))
            cur_inner_prod_avg = torch.mean(cur_inner_prod, dim=-1, keepdim=True)
            inner_prod_avgs.append(cur_inner_prod_avg)
        inner_prod_avgs = torch.cat(inner_prod_avgs, dim=-1)
        # # bz x nsmp x 2
        return inner_prod_avgs

    def generate_feature_cross_prod(self, nearest_k_idx, nearest_k_dist, features, fps_idx, r=0.03):
        bz, N, nsmp = features.size(0), features.size(1), nearest_k_idx.size(1)
        nearest_k_indicators = (nearest_k_dist <= r).float()
        print(torch.mean(torch.sum(nearest_k_indicators, dim=-1)))
        nearest_k_features = batched_index_select(features, indices=nearest_k_idx, dim=1)
        # # here only `pos` features are reasonable
        # # bz x nsmp x feat_dim
        features_sub = features.contiguous().view(bz * N, -1)[fps_idx].view(bz, nsmp, -1)
        features_minus = nearest_k_features - features_sub.unsqueeze(2)
        # # # bz x nsmp x feat_dim
        feat_x, feat_y, feat_z = features_sub[:, :, 0], features_sub[:, :, 1], features_sub[:, :, 2]
        nk_feat_x, nk_feat_y, nk_feat_z = nearest_k_features[:, :, :, 0], nearest_k_features[:, :, :, 1], nearest_k_features[:, :, :, 2]
        # # # feat_x.size = bz x nsmp
        cross_x = feat_y.unsqueeze(-1) * nk_feat_z - feat_z.unsqueeze(-1) * nk_feat_y
        cross_y = feat_z.unsqueeze(-1) * nk_feat_x - feat_x.unsqueeze(-1) * nk_feat_z
        cross_z = feat_x.unsqueeze(-1) * nk_feat_y - feat_y.unsqueeze(-1) * nk_feat_x
        # # bz x nsmp x k
        cross_prod = torch.cat([cross_x.unsqueeze(-1), cross_y.unsqueeze(-1), cross_z.unsqueeze(-1)], dim=-1)
        # # bz x nsmp x k x 3
        cross_prod_avg = torch.mean(cross_prod, dim=2)
        return cross_prod_avg

    def generate_feature_row_space(self, nearest_k_idx, nearest_k_dist, features, fps_idx, r=0.03, zero=False):
        bz, N, nsmp = features.size(0), features.size(1), nearest_k_idx.size(1)
        # # bz x nsmp x k x feat_dim
        # if zero:
        #     features = features - torch.mean(features, dim=2, keepdim=True)
        nearest_k_features = batched_index_select(features, indices=nearest_k_idx, dim=1)
        nearest_k_feat_split = [nearest_k_features[:, :, :, :3], nearest_k_features[:, :, :, 3:]]

        # # [pos, pos2]
        nearest_k_feat_split[1] = nearest_k_feat_split[0] + nearest_k_feat_split[1]

        if zero:
            nearest_k_feat_split[0] = nearest_k_feat_split[0] - torch.mean(nearest_k_feat_split[0], dim=2, keepdim=True)
            nearest_k_feat_split[1] = nearest_k_feat_split[1] - torch.mean(nearest_k_feat_split[1], dim=2, keepdim=True)

        nearest_k_feat_row_space = []
        for nearest_k_feat in nearest_k_feat_split:
            U, Sigma, VT = np.linalg.svd(nearest_k_feat.detach().cpu().numpy())
            # # if we randomly apply some sign change to let the model be aware that the row-space may
            # # bz x nsmp x 3 x 3
            # # # todo: test the reverse process...
            V = torch.from_numpy(VT).to(features.device).transpose(2, 3)
            V_det = torch.det(V)
            V_det_indicators = (V_det < -0.3).float()
            # print(torch.sum(V_det_indicators, dim=-1).mean())
            V_det_indicators = V_det_indicators.unsqueeze(-1).repeat(1, 1, 9).contiguous().view(bz, nsmp, 3, 3)
            V_det_indicators[:, :, :, :2] = 0.0
            V[V_det_indicators > 0.5] *= -1.0
            # V_det = torch.det(V)
            # V_det_indicators = (V_det < -0.3).float()
            # print(torch.sum(V_det_indicators, dim=-1).mean())
            # # V: the rotation of this space to the canonical space
            nearest_k_feat_row_space.append(V)
        # inner_product = []
        # for i in range(len(nearest_k_feat_row_space)):
        #     for j in range(len(nearest_k_feat_row_space)):
        #         if i == j:
        #             continue
        #         inner_product.append(torch.matmul(nearest_k_feat_row_space[i], nearest_k_feat_row_space[j].transpose(2, 3)))
        # nearest_k_feat_row_space = nearest_k_feat_row_space + inner_product
        matmul_prod = torch.matmul(nearest_k_feat_row_space[1], nearest_k_feat_row_space[0].transpose(2, 3))
        nearest_k_feat_row_space.append(matmul_prod)

        nearest_k_feat_row_space = [kk.contiguous().view(bz, nsmp, 9) for kk in nearest_k_feat_row_space]
        nearest_k_feat_row_space = torch.cat(nearest_k_feat_row_space, dim=-1) #

        return nearest_k_feat_row_space

    def forward(self, pos: torch.FloatTensor,
                masks: torch.FloatTensor,
                training=False,
                return_training=True,
                shape_seg_masks=None,
                pc1_af_rel=None,
                pc1_ori=None,
                batch_gt_transform_vec=None,
                r=0.03):
        masks = masks.float()
        if masks.size(1) == self.mask_dim:
            masks = masks.transpose(1, 2)
        else:
            assert masks.size(2) == self.mask_dim
        pos = pos.float()

        bz, N = pos.size(0), pos.size(1)
        nsmp = 256

        pos = pos.contiguous()

        x, global_x, pos = self.conv_uniunet(None, pos, return_global=True, use_ori_feat=True, r=None,
                conv_select_types=[0, 0, 0, 0, 0], relative_pos=True, edgeconv_interpolate=False, edgeconv_skip_con=False)
        # x, global_x, pos = self.conv_uniunet_further(x, pos, return_global=True, use_ori_feat=True, r=None,
        #         conv_select_types=[0, 0, 0, 0, 0], relative_pos=True, edgeconv_interpolate=False, edgeconv_skip_con=False)

        fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=self.cluster_seed_pts)  # total 512

        # pos_sub = pos.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
        # x_sub = x.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)

        seed_x = x.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, self.cluster_seed_pts, -1)
        seed_pos = pos.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, self.cluster_seed_pts, -1)

        x_sub = x

        pred_l = x_sub

        combined_x_seed = torch.cat(
            [x.unsqueeze(2).repeat(1, 1, self.cluster_seed_pts, 1), seed_x.unsqueeze(1).repeat(1, N, 1, 1)], dim=-1
        )

        grp_feat_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            combined_x_seed, self.grp_feat_trans_net
        )
        grp_feat_glb_x, _ = torch.max(grp_feat_x, dim=1, keepdim=True)
        grp_feat_glb_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            grp_feat_glb_x, self.grp_feat_trans_net_glb
        )
        grp_feat_combined_x = torch.cat(
            [grp_feat_x, grp_feat_glb_x.repeat(1, N, 1, 1)], dim=-1
        )
        grp_feat_combined_x = CorrFlowPredNet.apply_module_with_conv2d_bn(
            grp_feat_combined_x, self.grp_feat_combined_net
        )
        grp_feat_combined_x = grp_feat_combined_x.squeeze(-1)
        grp_feat_combined_x = torch.sigmoid(grp_feat_combined_x)
        segpred, confpred = self.mode_pred(grp_feat_combined_x)
        confpred = confpred.squeeze(-1)
        segpred = torch.softmax(segpred, dim=1)


        # seg_feat_cls = CorrFlowPredNet.apply_module_with_conv1d_bn(
        #     pred_l, self.cls_layers
        # )
        # seg_conf_cls = CorrFlowPredNet.apply_module_with_conv1d_bn(
        #     global_x, self.conv_pred_layers
        # )
        #
        # seg_feat_cls = torch.softmax(seg_feat_cls, dim=-1).transpose(1, 2)
        # seg_conf_cls = torch.sigmoid(seg_conf_cls)
        # print(seg_conf_cls.size())

        # masks_sub = masks.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
        # # print(pred_l.size(), x_sub.size(), x.size())
        # # pred_l = pred_l_sub
        # grp_feat = torch.cat(
        #     [pred_l.unsqueeze(2).repeat(1, 1, nsmp, 1), pred_l.unsqueeze(1).repeat(1, nsmp, 1, 1)], dim=-1
        # )
        # grp_local = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_feat, self.grp_feat_trans_net
        # )
        # # # 64 dim
        # grp_global, _ = torch.max(grp_local, dim=2, keepdim=True)
        # grp_global = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_global, self.grp_feat_trans_net_glb
        # )
        # grp_combined = torch.cat([grp_local, grp_global.repeat(1, 1, nsmp, 1)], dim=-1)
        # grp_combined = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_combined, self.grp_feat_combined_net
        # )
        #
        # grp_combined = grp_combined.squeeze(-1)
        # grp_combined = torch.sigmoid(grp_combined)
        #
        # # point_point_loss = -torch.mean(gt_sub_sub_pair_wise * torch.log(torch.clamp(grp_combined, min=1e-9)) +
        # #                     (1. - gt_sub_sub_pair_wise) * torch.log(torch.clamp(1. - grp_combined, min=1e-9)))
        # #
        # # losses.append(point_point_loss.detach().cpu().item())
        # # gt_l += point_point_loss
        # #
        # # # gt_l += -torch.mean(gt_sub_sub_pair_wise * torch.log(torch.clamp(grp_combined, min=1e-9)) +
        # # #                     (1. - gt_sub_sub_pair_wise) * torch.log(torch.clamp(1. - grp_combined, min=1e-9)))
        # #
        # # sim_part_sims = torch.sum(grp_combined * gt_sub_sub_pair_wise) / torch.sum(gt_sub_sub_pair_wise)
        # # unsim_part_sims = torch.sum(grp_combined * (1. - gt_sub_sub_pair_wise)) / torch.sum(1. - gt_sub_sub_pair_wise)
        # # # print(sim_part_sims)
        # # # print(unsim_part_sims)
        # # sims.append(sim_part_sims.detach().cpu().item())
        # # sims.append(unsim_part_sims.detach().cpu().item())
        # #
        # # # # add grouping losses
        # # # if len(losses) == 0:
        # # #     losses.append(gt_l.detach().cpu().item())
        # # # else:
        # # #     losses.append(gt_l.detach().cpu().item() - losses[-1])
        # #
        # #
        # # # # consturct the statistics for return
        # # statistics["sims"] = sims
        # # statistics["losses"] = losses
        #
        # segpred, confpred = self.mode_pred(grp_combined, nmask=10)
        # confpred = confpred.squeeze(-1)
        # confpred = torch.sigmoid(confpred)

        # seg_feat_sub = torch.softmax(segpred, dim=1)
        # pred_conf = confpred # .squeeze(-1)

        seg_feat_sub = segpred
        pred_conf = confpred

        # print(seg_feat_sub[0, :, :10].transpose(0, 1))

        gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)
        statistics = {"sim": [], "losses": []}

        return seg_feat_sub, fps_idx, pred_l, gt_l, pred_conf, statistics

    def get_gt_conf(self, momask):
        # bz x nmask x N?
        momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf

    def loss(self, pos: torch.FloatTensor, flow: torch.FloatTensor, masks: torch.FloatTensor,
                gt_rigid_trans: torch.FloatTensor = None,
             use_lm_pred=False, training=True):
        batch_momasks, batch_conf = self.get_gt_conf(masks)

        bz, N = pos.size(0), pos.size(1)
        nsmp = 256

        # seg_feat_sub, fps_idx, pred_l, gt_l, pred_l_gt, pred_mu_gt, pred_sigma_gt = self.forward(
        #     pos=pos, flow=flow, masks=batch_momasks, training=True)
        seg_feat_sub, fps_idx, pred_l, gt_l = self.forward(
            pos=pos, flow=flow, masks=batch_momasks, training=training, return_training=True)

        momasks_sub = batch_momasks.transpose(1, 2).contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
        # momasks_sub size = bz x nsmp x nmasks
        batch_momasks_sub, batch_conf_sub = self.get_gt_conf(momasks_sub)
        neg_iou_loss, loss_conf = iou(seg_feat_sub, batch_momasks_sub, batch_conf_sub)
        neg_iou_loss = (-neg_iou_loss).mean()
        return neg_iou_loss + gt_l, neg_iou_loss, gt_l
