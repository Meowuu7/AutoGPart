import torch
import torch.nn as nn
from .pnmem_cell import DynamicRNN
from .utils import farthest_point_sampling, batched_index_select, get_knn_idx
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
import torch.nn.functional as F
from .loss_model import ComputingGraphLossModel
# from .loss_model_v2 import ComputingGraphLossModel as ComputingGraphLossModel_v2
from model.utils import iou, mean_shift, labels_to_one_hot_labels
from torch.autograd import Variable

from .UNet_universal import UNet as UiUNet
from scipy.optimize import linear_sum_assignment
from .point_convolution_universal import LocalConvNet, EdgeConv
from .Poinetnet2 import PointnetPP
from .DGCNN import PrimitiveNet
from .mixstyle import MixStyle

import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PrimitiveFittingNet(nn.Module):
    def __init__(self, n_layers: int, feat_dims: list, n_samples: list, map_feat_dim: int,
                 landmark_type: str = "net", use_lm=True, stage=1, raw_feat_dim=1, lm_classes=10,
                 latent_type="clustering", clustering_feat_dim=64, args=None):
        super(PrimitiveFittingNet, self).__init__()

        ''' SET arguments '''
        self.args = args
        self.n_layers = n_layers
        self.feat_dims = feat_dims
        self.n_samples = n_samples
        self.use_lm = use_lm

        up_feat_dims = self.args.up_feat_dims.split(",")
        self.up_feat_dims = [int(ud) for ud in up_feat_dims]

        self.cluster_seed_pts = args.cluster_seed_pts

        self.xyz_dim = 3
        # maximum number of masks that can be processed in the model
        self.mask_dim = args.nmasks
        self.map_feat_dim = self.args.map_feat_dim
        # actually the mark of this model & exp
        self.landmark_type = landmark_type
        # representation dimension
        self.lm_dim = 64
        self.nsmp = 256
        self.raw_feat_dim = raw_feat_dim
        self.npoints = self.args.num_points
        self.nseeds = self.args.nseeds
        self.add_intermediat_loss = self.args.add_intermediat_loss

        self.local_unet_n_layers = self.args.local_unet_n_layers
        self.use_normal_loss = self.args.use_normal_loss

        self.mask_dim = args.nmasks
        self.in_feat_dim = self.args.in_feat_dim
        self.k = self.args.k
        self.use_spfn = self.args.use_spfn
        self.use_dgcnn = self.args.use_dgcnn
        self.use_mixstyle = self.args.mix_style

        self.num_points_sampled_for_seed = self.args.num_points_sampled_for_seed
        self.cls_backbone = self.args.cls_backbone
        self.with_normal = self.args.with_normal
        radius = self.args.radius.split(",")
        self.radius = [float(rr) for rr in radius]
        if not self.with_normal:
            self.in_feat_dim = self.in_feat_dim - 3 # exclude the normal feature vector
        ''' SET arguments '''

        ''' MixStyle transformation module '''
        if self.use_mixstyle:
            self.mixstyle = MixStyle()
        ''' MixStyle transformation module '''

        ''' 6-layer feature extraction backbone '''
        if self.use_dgcnn:
            args.dgcnn_out_dim = 128
            map_feat_dim = 128
            feat_dims[-1] = 1024
            self.conv_uniunet = PrimitiveNet(args)
        elif self.use_spfn:
            self.conv_uniunet = PointnetPP(in_feat_dim=self.in_feat_dim + 3, args=args)
            map_feat_dim = 128
            feat_dims[-1] = 1024
        else:
            self.conv_uniunet = UiUNet(
                feat_dims=feat_dims[:self.local_unet_n_layers],
                up_feat_dims=self.up_feat_dims[:self.local_unet_n_layers],
                n_samples=n_samples[:self.local_unet_n_layers],
                n_layers=self.local_unet_n_layers,
                in_feat_dim=self.in_feat_dim, # no extra input features except position information
                map_feat_dim=self.map_feat_dim,
                need_feat_map=True,
                k=self.k
            )

        ''' 6-layer feature extraction backbone '''

        ''' CONSTRUCT intermediate loss calculation network '''
        if self.add_intermediat_loss:
            if self.use_normal_loss:
                use_loss_model = ComputingGraphLossModel_v2
            else:
                use_loss_model = ComputingGraphLossModel
            self.intermediate_loss = use_loss_model(
                pos_dim=3, pca_feat_dim=64, in_feat_dim=map_feat_dim, pp_sim_k=16, r=0.03
            )
        ''' CONSTRUCT intermediate loss calculation network '''
        # self.conv_uniunet_bf_cluster = UiUNet(
        #     feat_dims=feat_dims[:self.local_unet_n_layers],
        #     n_samples=n_samples[:self.local_unet_n_layers],
        #     n_layers=self.local_unet_n_layers,
        #     in_feat_dim=map_feat_dim,  # no extra input features except position information
        #     map_feat_dim=map_feat_dim,
        #     need_feat_map=True
        # )

        # ###### Feature transformation for grouping features ######

        if not self.cls_backbone:
            ''' CONSTRUCT grouping score calculation network '''
            self.grp_feat_trans_net = construct_conv_modules(
                [map_feat_dim, map_feat_dim], n_in=map_feat_dim * 2, last_act=True
            )  # 64

            self.grp_feat_trans_net_glb = construct_conv_modules(
                [map_feat_dim, map_feat_dim], n_in=map_feat_dim, last_act=False
            )

            self.grp_feat_combined_net = construct_conv_modules(
                [map_feat_dim, map_feat_dim // 2, 1], n_in=map_feat_dim * 2, last_act=False, bn=False
            )
            ''' CONSTRUCT grouping score calculation network '''

            # self.glb_pred_l_pred_net = construct_conv1d_modules(
            #     [map_feat_dim * 2, map_feat_dim, map_feat_dim], n_in=map_feat_dim, last_act=False, bn=False,
            #     others_bn=False
            # )

            ''' CONSTRUCT score net '''
            self.score_net_point_conv2d = construct_conv_modules(
                [map_feat_dim, map_feat_dim], n_in=map_feat_dim, last_act=False, bn=True
            )
            self.score_net_cbd_conv2d = construct_conv_modules(
                [map_feat_dim, map_feat_dim], n_in=map_feat_dim * 2, last_act=False, bn=True
            )
            # TO score value
            self.score_net_part_conv1d = construct_conv1d_modules(
                [map_feat_dim, map_feat_dim // 2, 1], n_in=map_feat_dim, last_act=False, bn=True
            )
            ''' CONSTRUCT score net '''

        # self.cls_layers = construct_conv1d_modules(
        #     [map_feat_dim, map_feat_dim // 2, self.mask_dim], n_in=map_feat_dim, last_act=False, bn=False
        # )

        # todo: if using less masks?

        # self.mode_pred = DynamicRNN(nmasks=self.mask_dim)

        # nmasks = 24
        # todo: if we remove `bn` in the cls_layer?
        nmasks = self.args.pred_nmasks
        self.cls_layers = construct_conv1d_modules(
            [map_feat_dim, map_feat_dim, nmasks], n_in=map_feat_dim, last_act=False, bn=False
        )

        ''' CONSTRUCT primitive prediction layers '''
        self.n_primitives = self.args.n_primitives
        if self.args.with_primpred_loss:

            self.prm_pred_layers = construct_conv1d_modules(
                [map_feat_dim, map_feat_dim, self.n_primitives], n_in=map_feat_dim, last_act=False, bn=False
            )
        ''' CONSTRUCT primitive prediction layers '''

        ''' CONSTRUCT normal prediction layers '''
        if self.args.with_normalcst_loss:
            self.normal_pred_layers = construct_conv1d_modules(
                [map_feat_dim, map_feat_dim, 3], n_in=map_feat_dim, last_act=False, bn=False
            )
        ''' CONSTRUCT normal prediction layers '''

        ''' CONSTRUCT parameter prediction layers '''
        self.param_pred_dim = self.args.param_pred_dim
        if self.args.with_parampred_loss:

            self.param_pred_layers = construct_conv1d_modules(
                [map_feat_dim, map_feat_dim, self.param_pred_dim], n_in=map_feat_dim, last_act=False, bn=False
            )
        ''' CONSTRUCT parameter prediction layers '''

        ''' CONSTRUCT conf prediction layers '''
        if self.args.with_conf_loss:
            self.conf_pred = construct_conv1d_modules(
                [feat_dims[-1] // 2, feat_dims[-1] // 4, feat_dims[-1] // 8, nmasks], n_in=feat_dims[-1], last_act=False, bn=False
            )
        ''' CONSTRUCT conf prediction layers '''

        # self.seg_nmasks = self.args.seg_nmasks
        # self.mode_pred = DynamicRNN(nmasks=self.seg_nmasks)

        # glb_feat_dim = feat_dims[-1]
        # # glb_feat_dim = feat_dims[1] #
        # # conf prediction layers
        # self.conf_pred_layers = construct_conv1d_modules(
        #     [glb_feat_dim // 2, glb_feat_dim // 4, glb_feat_dim // 8, glb_feat_dim // 16, self.mask_dim], n_in=glb_feat_dim, last_act=False, bn=False
        # )
        # self.conf_pred_layers = construct_conv1d_modules(
        #     [glb_feat_dim, glb_feat_dim // 2, self.mask_dim],
        #     n_in=glb_feat_dim, last_act=False, bn=False
        # )

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

    def get_part_level_contrast_loss(self, x, nmasks, masks, point_part=False, part_part=False):
        bz, N = x.size(0), x.size(1)
        pred_l_expand = x.unsqueeze(1).repeat(1, nmasks, 1, 1)
        # # # bz x nmasks x N x feat_dim
        # # # masks.size = bz x N x nmasks
        masks_expand = masks.transpose(1, 2).unsqueeze(-1).repeat(1, 1, 1, x.size(-1))
        # # # masks_expand.size = bz x nmasks x N x feat_dim
        pred_l_expand[masks_expand < 0.5] = -1e9  # # mask out those features
        pred_part_level_features, _ = torch.max(pred_l_expand, dim=2)
        # print(pred_part_level_features[0, :10, :10])
        # # # bz x nmasks x feat_dim
        pred_part_level_features = CorrFlowPredNet.apply_module_with_conv1d_bn(
            pred_part_level_features, self.glb_pred_l_pred_net
        )

        loss = torch.zeros((1, ), dtype=torch.float32, device=masks.device)
        gt_conf = torch.sum(masks, dim=1)
        if point_part:
            # # print(pred_part_level_features[0, :10, :10])
            # # # labels.size = bz x N
            point_part_labels = torch.argmax(masks, dim=-1)
            # # # bz x N x nmasks
            pred_point_part_cosine_logits = torch.sum(x.unsqueeze(2) * pred_part_level_features.unsqueeze(1), dim=-1) / \
                                            (torch.clamp(torch.norm(x.unsqueeze(2), dim=-1, p=2) *
                                                         torch.norm(pred_part_level_features.unsqueeze(1), dim=-1, p=2),
                                                         min=1e-9))

            # # ###### Another method for calculating point-part loss and part-part loss ######
            gt_conf[gt_conf > 0.5] = 1.0
            gt_conf[gt_conf < 0.5] = 0.0
            num_valid_masks = torch.sum(gt_conf, dim=1)
            tot_point_part_contrast_loss = 0.0
            for i_bt in range(bz):
                cur_nmasks = int(num_valid_masks[i_bt].detach().item())
                cur_trunc_point_part_logits = pred_point_part_cosine_logits[i_bt, :, :] / 0.07
                # N x cur_nmasks
                cur_gt_conf = gt_conf[i_bt]
                # nmasks
                cur_gt_conf_expand = cur_gt_conf.unsqueeze(0).repeat(N, 1)
                cur_trunc_point_part_logits[cur_gt_conf_expand < 0.5] = -1e9
                cur_mask_labels = point_part_labels[i_bt]
                # print(cur_nmasks)
                # print(cur_mask_labels[:10])
                cur_point_part_loss = F.nll_loss(input=torch.log_softmax(cur_trunc_point_part_logits, dim=-1),
                                                 target=cur_mask_labels)
                tot_point_part_contrast_loss += cur_point_part_loss
            tot_point_part_contrast_loss = tot_point_part_contrast_loss / bz
            loss += tot_point_part_contrast_loss
        if part_part:
            gt_conf[gt_conf > 0.5] = 1.0
            gt_conf[gt_conf < 0.5] = 0.0
            num_valid_masks = torch.sum(gt_conf, dim=1)

            pred_part_part_cosine_logits = torch.sum(
                pred_part_level_features.unsqueeze(2) * pred_part_level_features.unsqueeze(1), dim=-1) / \
                                           (torch.clamp(
                                               torch.norm(pred_part_level_features.unsqueeze(2), dim=-1,
                                                          p=2) * torch.norm(
                                                   pred_part_level_features.unsqueeze(1), dim=-1, p=2), min=1e-9))
            tot_part_part_loss = 0.0
            for i_bt in range(bz):
                cur_nmasks = int(num_valid_masks[i_bt].detach().item())
                cur_gt_conf = gt_conf[i_bt]
                cur_gt_conf_expand_row = cur_gt_conf.unsqueeze(1).repeat(1, nmasks)
                cur_gt_conf_expand_col = cur_gt_conf.unsqueeze(0).repeat(nmasks, 1)
                cur_trunc_part_part_logits = pred_part_part_cosine_logits[i_bt, :, :]
                cur_gt_mask_labels = torch.eye(nmasks, dtype=torch.float32, device=x.device)
                cur_part_part_loss = -(1. - cur_gt_mask_labels) * torch.log(
                    torch.clamp(1. - cur_trunc_part_part_logits, min=1e-9))
                cur_part_part_loss[cur_gt_conf_expand_row < 0.5] = 0.0
                cur_part_part_loss[cur_gt_conf_expand_col < 0.5] = 0.0
                if cur_nmasks > 1:
                    cur_part_part_loss = torch.sum(cur_part_part_loss) / float(cur_nmasks * (cur_nmasks - 1))
                    tot_part_part_loss += cur_part_part_loss
            tot_part_part_loss = tot_part_part_loss / bz
            loss += tot_part_part_loss
        return loss

    def forward(
            self, pos: torch.FloatTensor,
            masks: torch.FloatTensor,
            inst_seg: torch.LongTensor,
            feats: {},
            conv_select_types=[0,0,0,0,0],
            point_feat_selection=4,
            point_geo_feat_selection=3,
            contrast_selection=[1,1,1]
        ):

        point_level_contrast, point_part_level_contrast, part_part_level_contrast = contrast_selection
        # masks = masks.float()
        # if masks.size(1) == self.mask_dim:
        #     masks = masks.transpose(1, 2)
        # else:
        #     assert masks.size(2) == self.mask_dim
        pos = pos.float()
        bz, N = pos.size(0), pos.size(1)
        # nsmp = 256

        fps_idx = None

        pos = pos.contiguous()

        ''' GET features '''
        feat = []
        for k in feats:
            feat.append(feats[k])
        feat = torch.cat(feat, dim=-1) if len(feat) > 0 else None
        ''' GET features '''

        # if self.cls_backbone:
        #     n_sampling = 4096
        #     fps_idx = farthest_point_sampling(pos=pos, n_sampling=n_sampling)
        #     pos_sub = pos.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_sampling, -1)
        #     feat_sub = feat.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_sampling, -1)
        #     distt = torch.sum((pos.unsqueeze(2) - pos_sub.unsqueeze(1)) ** 2, dim=-1)
        #     # minn_idx.size = bz x N
        #     minn_dist, minn_idx = torch.min(distt, dim=2)
        #     pos = pos_sub
        #     feat = feat_sub

        ''' GET feature embeddings '''
        if self.use_dgcnn:
            infeat = torch.cat([pos, feats["normals"]], dim=-1)
            x, type_per_point, normal_per_point = self.conv_uniunet(xyz=infeat, normal=feats["normals"], inds=None, postprocess=False)
        elif self.with_normal:
            x, global_x, pos = self.conv_uniunet(feat, pos, return_global=True, use_ori_feat=True, r=None,
                    conv_select_types=conv_select_types,
                                                 relative_pos=True, edgeconv_interpolate=False, edgeconv_skip_con=False,
                                                 radius=self.radius
                                                 )
        else:
            x, global_x, pos = self.conv_uniunet(None, pos, return_global=True, use_ori_feat=True, r=None,
                                                 conv_select_types=conv_select_types, relative_pos=True,
                                                 edgeconv_interpolate=False, edgeconv_skip_con=False,
                                                 radius=self.radius
                                                 )
        ''' GET feature embeddings '''

        if self.use_mixstyle:
            x = self.mixstyle(x)

        # x, global_x, pos = self.conv_uniunet_further(x, pos, return_global=True, use_ori_feat=True, r=None,
        #         conv_select_types=[0, 0, 0, 0, 0], relative_pos=True, edgeconv_interpolate=False, edgeconv_skip_con=False)

        # nsmp = 512
        # fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=512)  # total 512
        # mask_sub = masks.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nsmp, -1).contiguous()
        # x_sub = x.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nsmp, -1).contiguous()
        # pp_wise_labels = torch.matmul(mask_sub, mask_sub.transpose(1, 2))
        # pp_wise_sim = torch.sum(x_sub.unsqueeze(2) * x_sub.unsqueeze(1), dim=-1) / \
        #               torch.clamp(torch.norm(x_sub.unsqueeze(2), dim=-1, p=2, keepdim=False) * torch.norm(x_sub.unsqueeze(1), dim=-1, p=2, keepdim=False), min=1e-9)
        # point_point_loss = -torch.mean(pp_wise_labels * torch.log(torch.clamp(pp_wise_sim, min=1e-9)) +
        #                                (1. - pp_wise_labels) * torch.log(
        #                                 torch.clamp(1. - pp_wise_sim, min=1e-9)))
        #
        # losses = []
        # losses.append(point_point_loss.detach().cpu().item())
        # gt_l = point_point_loss
        #
        # xx = torch.cat([pos, x], dim=-1)
        # # todo: if we use two-layer edgeconv to aggregate point features?
        # x, global_x, pos = self.conv_uniunet_bf_cluster(
        #     x, pos, return_global=True, use_ori_feat=True, r=None,
        #     conv_select_types=[0, 0, 0, 0, 0], relative_pos=True, edgeconv_interpolate=False, edgeconv_skip_con=False
        # )
        #
        # x_cls = CorrFlowPredNet.apply_module_with_conv1d_bn(
        #     x, self.cls_layers
        # )
        # # bz x N x cls
        #
        # # global_x, _ = torch.max(global_x, dim=1, keepdim=True)
        # conf_pred = CorrFlowPredNet.apply_module_with_conv1d_bn(
        #     global_x, self.conf_pred_layers
        # )
        # # bz x N
        #
        # x_cls = torch.softmax(x_cls, dim=-1)
        # # possibility belonging to each class
        # segpred = x_cls.contiguous().transpose(1, 2)
        # # confpred = t
        # confpred = conf_pred.squeeze(1)

        ''' GET features & position for clustering seed points '''
        nseeds = self.nseeds

        # sub_pair_wise_sim = torch.sum((x.unsqueeze(2) * x_seed.unsqueeze(1)), dim=-1) / torch.clamp(
        #     torch.norm(x.unsqueeze(2), p=2, dim=-1) * torch.norm(x_seed.unsqueeze(1), p=2, dim=-1), min=1e-9
        # )
        #
        # gt_mask_mask_sub_logits = torch.matmul(masks, mask_seed.transpose(1, 2))
        # sig_sub_pair_wise_sim = torch.sigmoid(sub_pair_wise_sim)
        # point_point_loss = -torch.mean(gt_mask_mask_sub_logits * torch.log(torch.clamp(sig_sub_pair_wise_sim, min=1e-9)) +
        #                                (1. - gt_mask_mask_sub_logits) * torch.log(
        #     torch.clamp(1. - sig_sub_pair_wise_sim, min=1e-9)))


        # grp_feat = torch.cat(
        #     [x.unsqueeze(2).repeat(1, 1, nseeds, 1), x_seed.unsqueeze(1).repeat(1, N, 1, 1)], dim=-1
        # )
        # grp_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_feat, self.grp_feat_trans_net
        # )
        #
        # grp_global, _ = torch.max(grp_feat, dim=2, keepdim=True)
        # grp_global = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_global, self.grp_feat_trans_net_glb
        # )
        # grp_combined = torch.cat([grp_feat, grp_global.repeat(1, 1, nseeds, 1)], dim=-1)
        # grp_combined = CorrFlowPredNet.apply_module_with_conv2d_bn(
        #     grp_combined, self.grp_feat_combined_net
        # )
        # grp_combined = grp_combined.squeeze(-1)
        # grp_combined = torch.sigmoid(grp_combined)
        #
        losses = []
        sims = []
        gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)
        # gt_mask_mask_sub_logits = (inst_seg.unsqueeze(-1) == inst_seg_seed.unsqueeze(1)).float()
        # point_point_loss = -torch.mean(gt_mask_mask_sub_logits * torch.log(torch.clamp(grp_combined, min=1e-9)) +
        #                                (1. - gt_mask_mask_sub_logits) * torch.log(
        #     torch.clamp(1. - grp_combined, min=1e-9)))
        #
        # losses.append(point_point_loss.detach().cpu().item())
        # gt_l += point_point_loss
        #
        # sim_part_sims = torch.sum(sub_pair_wise_sim * gt_mask_mask_sub_logits) / torch.sum(gt_mask_mask_sub_logits)
        # unsim_part_sims = torch.sum(sub_pair_wise_sim * (1. - gt_mask_mask_sub_logits)) / torch.sum(1. - gt_mask_mask_sub_logits)
        # print(sim_part_sims)
        # print(unsim_part_sims)
        # sims.append(sim_part_sims.detach().cpu().item())
        # sims.append(unsim_part_sims.detach().cpu().item())

        # x = x / torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=1e-9)
        #

        # segpred = mean_shift(x, bandwidth=0.85)
        # segpred = labels_to_one_hot_labels(segpred, self.mask_dim)
        # segpred = segpred.float()
        # segpred = segpred.transpose(1, 2).contiguous()
        # confpred = torch.sum(segpred, dim=-1)

        #
        # # if self.stage == 1:
        #
        # grp_combined = torch.sigmoid(sub_pair_wise_sim)

        # if self.stage == 2:
        #     pred_l = x.detach()
        #     seed_pred_l = x_seed.detach()
        # else:
        pred_l = x

        # todo: add part-part, point-part loss calculation
        # point_part, part_part = True, True
        # point_part_loss = self.get_part_level_contrast_loss(
        #     x, self.mask_dim, masks, point_part=point_part, part_part=part_part)

        if self.add_intermediat_loss or self.args.with_normalcst_loss or self.args.with_primpred_loss or self.args.with_parampred_loss:
            n_samples_inter_loss = self.args.n_samples_inter_loss
            if n_samples_inter_loss < self.npoints:
                fps_idx = farthest_point_sampling(pos, n_sampling=n_samples_inter_loss)
            else:
                fps_idx = None

        ''' GET intermediate loss '''
        if self.add_intermediat_loss:
            matrices = {k: feats[k] for k in feats}
            if fps_idx is not None:
                pos_inter = pos.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples_inter_loss, -1).contiguous()
                pred_l_inter = pred_l.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples_inter_loss, -1).contiguous()
                masks_inter = masks.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples_inter_loss, -1).contiguous()
                for k in matrices:
                    matrices[k] = matrices[k].contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz,
                                                                                                          n_samples_inter_loss,
                                                                                                          -1).contiguous()
            else:
                pos_inter = pos
                pred_l_inter = pred_l
                masks_inter = masks


            intermediate_loss = self.intermediate_loss(
                pos=pos_inter, x=pred_l_inter, label=masks_inter, point_feat_selection=point_feat_selection,
                point_geo_feat_selection=point_geo_feat_selection,
                point_level_contrast=point_level_contrast,
                point_part_level_contrast=point_part_level_contrast,
                part_part_level_contrast=part_part_level_contrast,
                matrices=matrices
            )
            gt_l += intermediate_loss
        ''' GET intermediate loss '''
        confpred = None
        if self.cls_backbone:
            statistics = {}
            statistics['x'] = x
            ''' top-down segmentation '''
            segpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
                x, self.cls_layers
            )
            segpred = torch.clamp(segpred, min=-20, max=20)
            segpred = torch.softmax(segpred, dim=-1)

            # 3-interpolate?
            # pos_pos_sub_dist = torch.sum(())
            # segpred = batched_index_select(segpred, indices=minn_idx, dim=1)
            segpred = segpred.transpose(1, 2).contiguous()
            ''' top-down segmentation '''

            if self.args.with_primpred_loss or self.args.with_normalcst_loss or self.args.with_parampred_loss:
                if fps_idx is not None:
                    x_sub = x.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples_inter_loss, -1).contiguous()
                else:
                    x_sub = x

            ''' per-point primitive type prediction '''
            if self.args.with_primpred_loss:
                prmpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
                    x, self.prm_pred_layers
                )
                # prmpred = torch.softmax(prmpred, dim=-1)
                prmpred = torch.clamp(prmpred, min=-20, max=20)
                prmpred = prmpred.transpose(1, 2).contiguous()

                statistics["prmpred"] = prmpred
            ''' per-point primitive type prediction '''

            ''' normal prediction '''
            if self.args.with_normalcst_loss:
                normalpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
                    x, self.normal_pred_layers
                )
                normalpred = torch.clamp(normalpred, min=-1e9, max=1e9)
                normalpred = torch.div(
                    normalpred, torch.norm(normalpred, dim=-1, p=2, keepdim=True)
                )
                statistics["normalpred"] = normalpred
            ''' normal prediction '''

            ''' PARAMETER prediction '''
            if self.args.with_parampred_loss:
                parampred = CorrFlowPredNet.apply_module_with_conv1d_bn(
                    x, self.param_pred_layers
                )
                statistics["parampred"] = parampred
            ''' PARAMETER prediction '''

            ''' conf prediction '''
            if self.args.with_conf_loss:
                confpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
                    global_x, self.conf_pred
                )
                confpred = torch.clamp(confpred, min=-20, max=20)
                # confpred = torch.sigmoid(confpred).squeeze(1)
                confpred = confpred.squeeze(1)
                statistics["confpred"] = confpred
            ''' conf prediction '''

            masks_dim = masks.size(-1)
            # todo: test the effectiveness of aligning with masks
            if masks_dim > segpred.size(1):
                segpred = torch.cat(
                    [segpred, torch.zeros((bz, masks_dim - segpred.size(1), N), dtype=torch.float32, device=pos.device)],
                    dim=1
                )
                if self.args.with_conf_loss:
                    confpred = torch.cat(
                        [confpred, torch.zeros((bz, masks_dim - confpred.size(1)), dtype=torch.float32, device=pos.device)],
                        dim=-1
                    )
            # print(masks.size(), segpred.size(), confpred.size())
            assert masks_dim <= segpred.size(1)
            # assert masks_dim <= confpred.size(1)

            # gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)
            # statistics = {}

            # _, seg_prediction = torch.max(segpred, dim=1)
            # # bz x N
            # _, gt_seg = torch.max(masks, dim=-1)
            # bz x N
            # np.save("./pos.npy", pos.detach().cpu().numpy())
            # np.save("./seg_pred.npy", seg_prediction.detach().cpu().numpy())
            # np.save("./seg_gt.npy", gt_seg.detach().cpu().numpy())

            # fps_idx = None

        else:

            ''' Two choices --- sample centers via position information & sample centers via embedding features '''
            # fps_idx = farthest_point_sampling(pos=pos[:, :, :3], n_sampling=nseeds)
            fps_idx = farthest_point_sampling(pos=x, n_sampling=nseeds)
            x_seed = x.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nseeds, -1)
            # mask_seed = masks.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nseeds, -1)
            # inst_seg_seed = inst_seg.contiguous().view(bz * N)[fps_idx].contiguous().view(bz, nseeds)
            # pred_l = x
            seed_pred_l = x_seed
            ''' GET features & position for clustering seed points '''

            ''' GET point-point feature similarity '''
            grp_feat = torch.cat(
                [pred_l.unsqueeze(2).repeat(1, 1, nseeds, 1), seed_pred_l.unsqueeze(1).repeat(1, N, 1, 1)], dim=-1
            )
            grp_feat = CorrFlowPredNet.apply_module_with_conv2d_bn(
                grp_feat, self.grp_feat_trans_net
            )

            grp_global, _ = torch.max(grp_feat, dim=2, keepdim=True)
            grp_global = CorrFlowPredNet.apply_module_with_conv2d_bn(
                grp_global, self.grp_feat_trans_net_glb
            )
            grp_combined = torch.cat([grp_feat, grp_global.repeat(1, 1, nseeds, 1)], dim=-1)
            grp_combined = CorrFlowPredNet.apply_module_with_conv2d_bn(
                grp_combined, self.grp_feat_combined_net
            )
            ''' GET point-point feature similarity '''

            # grp_combined = CorrFlowPredNet.apply_module_with_conv1d_bn(
            #     x, self.cls_layers
            # )
            # segpred = torch.softmax(segpred, dim=-1)
            # segpred = segpred.transpose(1, 2).contiguous()

            # todo: effectiveness comparision between this grp_combined calculation and the network based calculation
            # grp_combined = torch.sum(pred_l.unsqueeze(2) * seed_pred_l.unsqueeze(1), dim=-1) / \
            #                (torch.norm(pred_l.unsqueeze(2), dim=-1, p=2) * torch.norm(seed_pred_l.unsqueeze(1), dim=-1, p=2) + 1e-9)

            grp_combined = grp_combined.squeeze(-1)
            grp_combined = torch.clamp(grp_combined, min=-20, max=20)

            ''' GET score for each mask '''
            _, sampled_pts_for_mask_idx = torch.topk(
                input=grp_combined.transpose(1, 2).contiguous(), dim=2, k=self.num_points_sampled_for_seed)
            # bz x nseeds x num_points_sampled_for_seed x k
            point_feat_expand = batched_index_select(values=x, indices=sampled_pts_for_mask_idx)
            point_feat_expand = point_feat_expand.contiguous().transpose(1, 2).contiguous()
            # grp_combined.size = bz x N x nseeds
            # x.size = bz x N x feat_dim
            # point_feat_expand = x.unsqueeze(2).repeat(1, 1, nseeds, 1)
            # point_feat_expand = point_feat_expand * grp_combined.unsqueeze(-1)
            # bz x N x nseeds x k
            point_feat_expand = CorrFlowPredNet.apply_module_with_conv2d_bn(
                point_feat_expand, self.score_net_point_conv2d
            )
            mask_feat, _ = torch.max(point_feat_expand, dim=1, keepdim=True)
            combined_feat_expand = CorrFlowPredNet.apply_module_with_conv2d_bn(
                torch.cat(
                    [point_feat_expand, mask_feat.repeat(1, self.num_points_sampled_for_seed, 1, 1)],
                    dim=-1
                ),
                self.score_net_cbd_conv2d
            )
            combined_feat, _ = torch.max(combined_feat_expand, dim=1, keepdim=False)
            # bz x nseeds x 1
            mask_scores = CorrFlowPredNet.apply_module_with_conv1d_bn(
                combined_feat, self.score_net_part_conv1d
            )
            # not necessary above 0
            mask_scores = mask_scores.squeeze(-1)
            mask_scores = torch.sigmoid(mask_scores)
            # print(mask_scores.size())
            ''' GET score for each mask '''

            grp_combined = torch.sigmoid(grp_combined)
            # grp_combined = torch.softmax(grp_combined, dim=-1)
            ''' GET point-point feature similarity '''
            # grp_combined = torch.softmax(grp_combined, dim=-1)

            # sim_part_sims = torch.sum(grp_combined * gt_mask_mask_sub_logits) / torch.sum(gt_mask_mask_sub_logits)
            # unsim_part_sims = torch.sum(grp_combined * (1. - gt_mask_mask_sub_logits)) / torch.sum(
            #     1. - gt_mask_mask_sub_logits)
            # # print(sim_part_sims)
            # # print(unsim_part_sims)
            # sims.append(sim_part_sims.detach().cpu().item())
            # sims.append(unsim_part_sims.detach().cpu().item())

            # point_point_loss = -torch.mean(
            #     gt_mask_mask_sub_logits * torch.log(torch.clamp(grp_combined, min=1e-9)) +
            #     (1. - gt_mask_mask_sub_logits) * torch.log(
            #         torch.clamp(1. - grp_combined, min=1e-9)))
            # todo: some exps to verify the clustering results when using 0.8 * logits etc...

            ''' GET point-point feature similarity '''
            # pos_pp_loss = -torch.sum(gt_mask_mask_sub_logits * torch.log(torch.clamp(grp_combined, min=1e-9))) / \
            #     torch.clamp(torch.sum(gt_mask_mask_sub_logits), min=1e-9)
            # neg_pp_loss = -torch.sum((1. - gt_mask_mask_sub_logits) * torch.log(torch.clamp(1. - grp_combined, min=1e-9))) / \
            #     torch.clamp(torch.sum(1. - gt_mask_mask_sub_logits), min=1e-9)
            ''' GET point-point similarity loss '''

            # log_pi = torch.log(torch.clamp(grp_combined, min=1e-9))
            # log_pi = torch.log(torch.clamp(grp_combined, min=1e-9))
            # logits_mult_log_logits = torch.sum(grp_combined.unsqueeze(2) * log_pi.unsqueeze(1), dim=-1)
            # logits_mult_self_log_logits = torch.sum(grp_combined * log_pi, dim=-1)
            # pair_wise_cr = -logits_mult_log_logits + logits_mult_self_log_logits.unsqueeze(2)
            # # pair_wise_cr
            # pair_wise_gt_sim = torch.matmul(masks, masks.transpose(1, 2))
            # pos_pair_wise_cr = torch.sum(pair_wise_gt_sim * pair_wise_cr) / torch.sum(pair_wise_gt_sim)
            # neg_pair_wise_cr = torch.sum((1. - pair_wise_gt_sim) * pair_wise_cr) / torch.sum(1. - pair_wise_gt_sim)
            # gt_l = 2.25 * pos_pair_wise_cr - neg_pair_wise_cr

            # gt_l = point_point_loss # point point loss
            # losses.append(pos_pp_loss.detach().item())
            # losses.append(neg_pp_loss.detach().item())
            # losses.append(point_part_loss.detach().item())

            # ### END contrastive based loss ####

            ''' GET grouping scores '''

            pred_grp_matching_score = torch.matmul(grp_combined.transpose(1, 2), masks)
            # nseeds x nmasks

            pred_gt_iou = torch.div(
                pred_grp_matching_score, torch.clamp(torch.sum(grp_combined, dim=1, keepdim=False).unsqueeze(-1) + torch.sum(masks, dim=1, keepdim=True) - pred_grp_matching_score, min=1e-5)
            )

            ''' NMS '''
            gt_conf = torch.sum(masks, dim=1)
            gt_conf[gt_conf > 0.5] = 1.0
            gt_conf[gt_conf < 0.5] = 0.0
            curnmasks = torch.sum(gt_conf, dim=-1)
            curnmasks = curnmasks.detach().cpu().numpy()
            curnmasks = curnmasks.astype('int32')
            from .utils import non_maximum_suppression
            matching_idx = np.zeros((bz, masks.size(-1), 2)).astype('int32')

            ''' GET matching loss between predicted scores and gt scores '''
            pred_gt_matching_scores = torch.max(pred_gt_iou, dim=-1)[0]
            high_matching_socres = 0.75
            low_matching_scores = 0.25
            pred_gt_matching_scores[pred_gt_matching_scores > high_matching_socres] = 1.0
            pred_gt_matching_scores[pred_gt_matching_scores < low_matching_scores] = 0.0
            zzz = pred_gt_matching_scores - low_matching_scores
            zzz[zzz < 0.0] = 0.0
            zzz2 = high_matching_socres - pred_gt_matching_scores
            zzz2[zzz2 < 0.0] = 0.0
            zzz = zzz * zzz2
            pred_gt_matching_scores[zzz >= 0] = \
                (pred_gt_matching_scores[zzz >= 0] - low_matching_scores) / (high_matching_socres - low_matching_scores)
            gt_score_matching_loss = -(pred_gt_matching_scores * torch.log(torch.clamp(mask_scores, min=1e-9)) + (1. - pred_gt_matching_scores) * torch.log(torch.clamp(1. - mask_scores, min=1e-9)))
            # gt_score_matching_loss = torch.sum(gt_score_matching_loss * gt_conf, dim=-1) / (torch.sum(gt_conf, dim=-1) + 1e-9)
            gt_score_matching_loss = gt_score_matching_loss.mean()
            ''' GET matching loss between predicted scores and gt scores '''

            scores = mask_scores
            pred_pred_matching_score = torch.matmul(grp_combined.transpose(1, 2), grp_combined)
            pred_pred_iou = torch.div(
                pred_pred_matching_score, torch.clamp(
                    torch.sum(grp_combined, dim=1, keepdim=False).unsqueeze(-1) + torch.sum(grp_combined, dim=1,
                                                                                            keepdim=True) - pred_pred_matching_score,
                    min=1e-5)
            )
            confpred = torch.zeros((bz, masks.size(-1)), device=pos.device, dtype=torch.float32)
            gt_logits_confpred = torch.zeros((bz, masks.size(-1)), device=pos.device, dtype=torch.float32)
            for i in range(bz):
                cur_pred_pred_ious = pred_pred_iou[i]
                cur_pred_scores = scores[i]
                thd = 0.3
                pick = non_maximum_suppression(cur_pred_pred_ious, cur_pred_scores, thd)
                pick = torch.from_numpy(pick).to(pos.device).long()
                cur_grp_combined = grp_combined[i, :, pick]
                # cur_grp_combined_normalized = cur_grp_combined / (torch.sum(cur_grp_combined, dim=-1, keepdim=True) + 1e-8)
                # cur_grp_combined_normalized = cur_grp_combined / (torch.sum(cur_grp_combined, dim=-1, keepdim=True) + 1e-8)
                cur_grp_combined_normalized = torch.softmax(cur_grp_combined, dim=-1)
                cur_matching_score = torch.matmul(cur_grp_combined_normalized.transpose(0, 1), masks[i, :, :])

                # cur_matching_score = pred_grp_matching_score[i, pick, :]
                cur_pred_iou = torch.div(cur_matching_score, torch.clamp(
                    torch.sum(cur_grp_combined_normalized, dim=0, keepdim=False).unsqueeze(-1) + torch.sum(masks[i], dim=0,
                                                                                                keepdim=True) - cur_matching_score,
                    min=1e-5))
                curnmask = int(curnmasks[i].item())
                row_ind, col_ind = linear_sum_assignment(999. - cur_pred_iou.detach().cpu().numpy())
                curnmask = min(curnmask, pick.size(0))
                # curnmasks[i] = curnmask
                matching_idx[i, :curnmask, 0] = row_ind[:curnmask]
                matching_idx[i, :curnmask, 1] = col_ind[:curnmask]
                confpred[i, :curnmask] = 1.0
                gt_logits_confpred[i, curnmask: ] = -1e5

            matching_idx = torch.from_numpy(matching_idx).to(pos.device).long()
            segpred = batched_index_select(values=grp_combined.transpose(1, 2), indices=matching_idx[:, :, 0], dim=1)
            # momask_pred = batched_index_select(values=masks.transpose(1, 2), indices=matching_idx[:, :, 1], dim=1)
            # bz x
            # for i, curnmask in enumerate(curnmasks):
            #     # segpred[i, curnmask:, :] = 0.0
            #     # momask_pred[i, curnmask:, :] = 0.0
            #     confpred[i, :curnmask] = 1.0

            segpred = segpred * confpred.unsqueeze(-1)
            # momask_pred = momask_pred * confpred.unsqueeze(-1)
            # segpred = segpred + gt_logits_confpred.unsqueeze(-1)
            # segpred = torch.softmax(segpred, dim=1) # softmax for probability logits
            segpred = segpred / (torch.sum(segpred, dim=1, keepdim=True) + 1e-8)

            # pred_matching_scores = torch.sum(segpred * momask_pred, dim=-1)
            # pred_matching_scores = pred_matching_scores * confpred
            # pred_iou = torch.div(
            #     pred_matching_scores,
            #     torch.clamp(torch.sum(segpred, dim=-1) + torch.sum(momask_pred, dim=-1) - pred_matching_scores, min=1e-9)
            # )
            # pred_iou = pred_iou * confpred
            # pred_iou = torch.sum(pred_iou, dim=-1) / torch.clamp(torch.sum(confpred, dim=-1), min=1e-9)
            # pred_iou = torch.mean(pred_iou)
            #### NMS ####



            # #### Not NMS ####
            # pred_gt_iou_np = 999. - pred_gt_iou.detach().cpu().numpy()
            # matching_idx = np.zeros((bz, masks.size(-1), 2)).astype('int32')
            # gt_conf = torch.sum(masks, dim=1)
            # gt_conf[gt_conf > 0.5] = 1.0
            # gt_conf[gt_conf < 0.5] = 0.0
            # curnmasks = torch.sum(gt_conf, dim=-1)
            # curnmasks = curnmasks.detach().cpu().numpy()
            # curnmasks = curnmasks.astype('int32')
            # for i, curnmask in enumerate(curnmasks):
            #     # print(curnmask.shape)
            #     curnmask = int(curnmask)
            #     row_ind, col_ind = linear_sum_assignment(pred_gt_iou_np[i, :, :])
            #     matching_idx[i, :curnmask, 0] = row_ind[:curnmask]
            #     matching_idx[i, :curnmask, 1] = col_ind[:curnmask]
            # matching_idx = torch.from_numpy(matching_idx).to(pos.device).long()
            # segpred = batched_index_select(values=grp_combined.transpose(1, 2), indices=matching_idx[:, :, 0], dim=1)
            # confpred = torch.zeros((bz, masks.size(-1)), device=pos.device, dtype=torch.float32)
            # for i, curnmask in enumerate(curnmasks):
            #     segpred[i, curnmask:, :] = 0.0
            #     confpred[i, :curnmask] = 1.0
            # # confpred =
            # segpred = segpred / (torch.sum(segpred, dim=1, keepdim=True) + 1e-8)
            # #### Not NMS ####

            # segpred = segpred.transpose(1, 2).contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nsmp, -1).contiguous().transpose(1, 2)
            # print(segpred.size(), confpred.size())

            # nseeds x nmasks
            # matched_iou, matched_idx = torch.max(pred_gt_iou, dim=-1)
            # # todo: whether we can use `score` to help with the process?
            # # theta_h, theta_l = 0.75, 0.25
            # # gt_score = torch.ones_like(matched_iou)
            # # gt_score[matched_iou >= theta_h] = 1.0
            # # gt_score[matched_iou < theta_l] = 0.0
            # # gt_score[theta_l <= matched_iou < theta_h] = (matched_iou - theta_l) / (theta_h - theta_l)
            # back_mask = (matched_iou >= 0.5).float()
            # back_matched_iou = matched_iou * back_mask
            # neg_iou_loss = -torch.div(
            #     torch.sum(back_matched_iou, dim=-1), torch.clamp(torch.sum(back_mask, dim=-1), min=1e-9)
            # )
            # neg_iou_loss = torch.mean(neg_iou_loss)

            # matching_idx = torch.from_numpy(matching_idx).to(pos.device).long()
            # matched_iou = pred_gt_iou[:, matching_idx[:, :, 0], matching_idx[:, :, 1]]
            # matched_iou_row = batched_index_select(pred_gt_iou, matching_idx[:, :, 0], dim=1)
            # matched_iou_col = batched_index_select(matched_iou_row, matching_idx[:, :, 1].unsqueeze(1).repeat(1, matched_iou_row.size(1), 1), dim=2)
            # matched_iou = torch.zeros((bz, ))
            # print(matched_iou.size())
            # avg_matched_iou = torch.sum(gt_conf * matched_iou, dim=-1) / torch.clamp(torch.sum(gt_conf, dim=-1), min=1e-9)
            # avg_matched_iou = torch.mean(avg_matched_iou)
            # neg_iou_loss = -avg_matched_iou

            # valid_matched_iou_mash = torch.zeros_like(pred_gt_iou)
            # for i, curnmask in enumerate(curnmasks):
            #     for j in range(curnmask):
            #         row_idx, col_idx = matching_idx[i, j, 0].item(), matching_idx[i, j, 1].item()
            #         valid_matched_iou_mash[i, row_idx, col_idx] = 1.0
            # # bz x nseeds x nmasks
            # valid_pred_gt_iou = valid_matched_iou_mash * pred_gt_iou
            # # valid_pred_gt_iou = valid_pred_gt_iou / (torch.sum(valid_pred_gt_iou, dim=-1, keepdim=True) + 1e-8)
            # avg_valid_iou_value = torch.sum(torch.sum(valid_pred_gt_iou, dim=-1), dim=-1) / \
            #     torch.clamp(torch.sum(torch.sum(valid_matched_iou_mash, dim=-1), dim=-1), min=1e-9)
            # avg_valid_iou_value = torch.mean(avg_valid_iou_value)
            # neg_iou_loss = -avg_valid_iou_value
            #
            # # valid_pred = torch.mean(torch.mean(back_mask, dim=-1), dim=0)
            # sims.append(-neg_iou_loss.detach().cpu().item())
            #
            # print(avg_valid_iou_value.detach().cpu().item())

            # print(pred_iou.detach().cpu().item())
            # neg_iou_loss = -pred_iou

            # gt_l += pos_pp_loss + neg_pp_loss + point_part_loss + neg_iou_loss
            # gt_l += neg_iou_loss # + pos_pp_loss # + neg_pp_loss
            # gt_l += 4 * pos_pp_loss + neg_pp_loss + gt_score_matching_loss
            gt_l += gt_score_matching_loss # + pos_pp_loss + neg_pp_loss

            # if self.stage == 2:
                # grp_combined = grp_combined.detach()

            # grp_combined = gt_mask_mask_sub_logits
            # gt_mask_mask_sub_logits

            ''' GET similarity statistics '''
            # sim_part_sims = torch.sum(grp_combined * gt_mask_mask_sub_logits) / torch.sum(gt_mask_mask_sub_logits)
            # unsim_part_sims = torch.sum(grp_combined * (1. - gt_mask_mask_sub_logits)) / torch.sum(
            #     1. - gt_mask_mask_sub_logits)
            # # print(sim_part_sims)
            # # print(unsim_part_sims)
            # sims.append(sim_part_sims.detach().cpu().item())
            # sims.append(unsim_part_sims.detach().cpu().item())
            ''' GET similarity statistics '''
            #
            # sims.append(pos_pair_wise_cr.detach().cpu().item())
            # sims.append(neg_pair_wise_cr.detach().cpu().item())

            # grp_combined = grp_combined.detach()

            # sub_grp_combined = grp_combined.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nsmp,
            #                                                                                             -1).contiguous()
            #
            # gt_mask_sub_mask_sub_logits = torch.matmul(mask_seed, mask_seed.transpose(1, 2))
            # segpred = mean_shift(sub_grp_combined, bandwidth=0.85)
            # segpred = labels_to_one_hot_labels(segpred, self.mask_dim)
            # segpred = segpred.float()
            # segpred = segpred.transpose(1, 2).contiguous()
            # confpred = torch.sum(segpred, dim=-1)

            # #### real mode prediction ####
            # segpred, confpred = self.mode_pred(sub_grp_combined)
            # confpred = confpred.squeeze(-1)
            # confpred = torch.sigmoid(confpred)
            # segpred = torch.softmax(segpred, dim=1)

            # segpred = segpred / torch.clamp(torch.sum(segpred, dim=1, keepdim=True), min=1e-9)

            # grp_combined = torch.from_numpy(grp_combined.detach().cpu().numpy()).to(pos.device)
            # grp_combined[grp_combined < 0.5] = 0.0
            # grp_combined[grp_combined > 0.5] = 1.0

            # grp_combined = grp_combined / torch.clamp(torch.norm(grp_combined, p=2, dim=-1, keepdim=True), min=1e-9)



            # if not training:
            #     sub_grp_combined = grp_combined.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, nsmp,
            #                                                                                                 -1).contiguous()
            #
            #     # _, spectral_embedding = torch.lobpcg(sub_grp_combined, k=64, niter=10)
            #     # spectral_embedding = spectral_embedding / torch.clamp(
            #     #     torch.norm(spectral_embedding, dim=-1, p=2, keepdim=True), min=1e-9)
            #
            #     spectral_embedding = sub_grp_combined
            #
            #     segpred = mean_shift(spectral_embedding, bandwidth=0.85)
            #     segpred = labels_to_one_hot_labels(segpred, self.mask_dim)
            #     # segpred.float()
            #     segpred = segpred.float()
            #     segpred = segpred.transpose(1, 2).contiguous()
            #     confpred = torch.sum(segpred, dim=-1)
            # else:
            #     #### fake dummy variables ####
            #     segpred = mask_seed.clone()
            #     segpred = segpred.transpose(1, 2)
            #     confpred = torch.ones((bz, self.mask_dim), dtype=torch.float, device=pos.device)
            #     segpred = segpred.detach()
            #     confpred = confpred.detach()

            # if self.stage == 2:
            #     segpred = segpred.detach()
            #     confpred = confpred.detach()

            # seg_feat_sub = torch.softmax(segpred, dim=1)
            # pred_conf = confpred  # .squeeze(-1)

            # gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)

            # gt_mask_sub_mask_sub_logits[gt_mask_sub_mask_sub_logits > 0.5] = 0.5
            # gt_mask_sub_mask_sub_logits[gt_mask_sub_mask_sub_logits < 0.5] = 0.5
            # spectral_embedding = gt_mask_sub_mask_sub_logits
            #
            # segpred = mean_shift(spectral_embedding, bandwidth=0.85)
            # segpred = labels_to_one_hot_labels(segpred, self.mask_dim)
            # # segpred.float()
            # segpred = segpred.float()
            # segpred = segpred.transpose(1, 2).contiguous()
            # confpred = torch.sum(segpred, dim=-1)

            # if segpred.size(1) < self.mask_dim:
            #     tile_seg = torch.zeros((bz, self.mask_dim - segpred.size(1), N), device=pos.device, dtype=torch.float32)
            #     segpred = torch.cat(
            #         [segpred, tile_seg], dim=1
            #     )
            #     tile_conf = torch.zeros((bz, self.mask_dim - self.seg_nmasks), device=pos.device, dtype=torch.float32)
            #     confpred = torch.cat(
            #         [confpred, tile_conf], dim=1
            #     )

            # statistics = {"sim": sims, "losses": losses}
            # statistics = {"sim": sims, "iou": pred_iou}
            statistics = {"sim": sims}

        return segpred, gt_l, confpred, statistics, fps_idx

    def get_gt_conf(self, momask):
        # bz x nmask x N?
        momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf
