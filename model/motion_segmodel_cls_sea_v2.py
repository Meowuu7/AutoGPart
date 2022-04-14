import torch
import torch.nn as nn
from .pnmem_cell import DynamicRNN
from .utils import farthest_point_sampling, batched_index_select, get_knn_idx
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
import torch.nn.functional as F
# from .loss_model import ComputingGraphLossModel
# from .loss_model_v2 import ComputingGraphLossModel as ComputingGraphLossModel_v2
# from .loss_model_v3 import ComputingGraphLossModel as ComputingGraphLossModel_v3
# from .loss_model_v4 import ComputingGraphLossModel as ComputingGraphLossModel_v4
from .loss_model_v5 import ComputingGraphLossModel as ComputingGraphLossModel_v5
from model.utils import iou, mean_shift, labels_to_one_hot_labels
from torch.autograd import Variable

from .UNet_universal import UNet as UiUNet
from scipy.optimize import linear_sum_assignment
from .point_convolution_universal import LocalConvNet, EdgeConv
from .Poinetnet2 import PointnetPP
from .DGCNN import PrimitiveNet
from .UniConvNet import UniConvNet
from .mixstyle import MixStyle
from model.model_util import estimate_normals

import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class InstSegNet(nn.Module):
    def __init__(self, n_layers: int, feat_dims: list, n_samples: list, map_feat_dim: int,
                 landmark_type: str = "net", use_lm=True, stage=1, raw_feat_dim=1, lm_classes=10,
                 latent_type="clustering", clustering_feat_dim=64, args=None):
        super(InstSegNet, self).__init__()

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
        self.use_mixstyle = self.args.mix_style

        self.local_unet_n_layers = self.args.local_unet_n_layers
        self.use_normal_loss = self.args.use_normal_loss

        self.mask_dim = args.nmasks
        self.in_feat_dim = self.args.in_feat_dim
        self.k = self.args.k
        self.use_spfn = self.args.use_spfn
        self.use_uniconv = self.args.use_uniconv

        self.skip_global = self.args.skip_global
        self.in_model_loss_model = self.args.in_model_loss_model

        self.num_points_sampled_for_seed = self.args.num_points_sampled_for_seed
        self.cls_backbone = self.args.cls_backbone
        self.with_normal = self.args.with_normal
        radius = self.args.radius.split(",")
        self.radius = [float(rr) for rr in radius]

        self.use_dgcnn = self.args.use_dgcnn
        if not self.with_normal:
            self.in_feat_dim = self.in_feat_dim - 3 # exclude the normal feature vector
        ''' SET arguments '''

        self.rep_dim = 128
        if self.use_uniconv:
            self.dgcnn_out_dim = 128
            map_feat_dim = 128
            feat_dims[-1] = 1024
            self.rep_dim = map_feat_dim

            self.conv_uniunet = UniConvNet(in_feat_dim=self.in_feat_dim + 3, args=args)
        elif self.use_dgcnn:
            print("Using DGCNN...")
            args.dgcnn_out_dim = 128
            args.dgcnn_in_feat_dim = 6 if self.args.input_normal else 3
            self.conv_uniunet = PrimitiveNet(args)
            map_feat_dim = 128
            feat_dims[-1] = 1024
        elif self.use_spfn:
            ''' A classical pointnetpp backbone '''
            self.conv_uniunet = PointnetPP(in_feat_dim=self.in_feat_dim + 3, args=args)
            map_feat_dim = 128
            feat_dims[-1] = 1024
            ''' A classical pointnetpp backbone '''
        else:
            ''' 6-layer feature extraction backbone '''
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
        if self.in_model_loss_model: # and self.add_intermediat_loss:
            self.intermediate_loss = ComputingGraphLossModel_v5(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                  lr_scaler=args.lr_scaler, init_lr=float(args.init_lr),
                                                  weight_decay=float(args.weight_decay),
                                                  loss_model_save_path=args.loss_model_save_path,
                                                  in_rep_dim=self.rep_dim,
                                                  nn_inter_loss=args.nn_inter_loss,
                                                  args=args
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

        ''' MixStyle transformation module '''
        if self.use_mixstyle:
            self.mixstyle = MixStyle()
        ''' MixStyle transformation module '''

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


        ''' SET number of masks for segmentations to predict '''
        nmasks = self.args.pred_nmasks
        ''' SET classification layers for top-down segmentation proposal '''
        self.cls_layers = construct_conv1d_modules(
            [map_feat_dim, map_feat_dim, nmasks], n_in=map_feat_dim, last_act=False, bn=False
        )
        ''' SET classification layers for top-down segmentation proposal '''

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

        ''' CONSTRUCT rotationa & transolation prediction layers '''
        if self.args.with_rt_loss:
            #### 6 dims for rotation matrix + 3 dims for translation vector ####
            self.rt_pred = construct_conv1d_modules(
                [map_feat_dim, map_feat_dim, 9], n_in=map_feat_dim, last_act=False, bn=False
            )
        ''' CONSTRUCT rotationa & transolation prediction layers '''

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

    # R.size = bz x N x 3 x 3
    def get_rotation_matrix(self, Rs):
        a1s = Rs[:, :, :, 0].unsqueeze(-1)
        a2s = Rs[:, :, :, 1].unsqueeze(-1)
        b1s = a1s / torch.norm(a1s, dim=2, p=2, keepdim=True)
        b2s = a2s - (torch.sum(b1s * a2s, dim=2, keepdim=True)) * b1s
        b2s = b2s / torch.norm(b2s, dim=2, p=2, keepdim=True)
        b3s = torch.zeros_like(b2s)
        b3s[:, :, 0, 0] = b1s[:, :, 1, 0] * b2s[:, :, 2, 0] - b1s[:, :, 2, 0] * b2s[:, :, 1, 0]
        b3s[:, :, 1, 0] = -(b1s[:, :, 0, 0] * b2s[:, :, 2, 0] - b1s[:, :, 2, 0] * b2s[:, :, 0, 0])
        b3s[:, :, 2, 0] = b1s[:, :, 0, 0] * b2s[:, :, 1, 0] - b1s[:, :, 1, 0] * b2s[:, :, 0, 0]
        # tb1, tb2, tb3 = b1s[0, 0, :, 0], b2s[0, 0, :, 0], b3s[0, 0, :, 0]
        Rs = torch.cat([b1s, b2s, b3s], dim=-1)
        # print(torch.sum((torch.det(Rs) < 0).long()))
        return Rs

    # R.size = N x 3 x 3
    def get_rotation_matrix_dim_2(self, Rs):
        a1s = Rs[:, :, 0].unsqueeze(-1)
        a2s = Rs[:, :, 1].unsqueeze(-1)
        b1s = a1s / torch.norm(a1s, dim=1, keepdim=True)
        b2s = a2s - (torch.sum(b1s * a2s, dim=1, keepdim=True) / torch.norm(a2s, dim=1, keepdim=True)) * b1s
        b2s = b2s / torch.norm(b2s, dim=1, keepdim=True)
        b3s = torch.zeros_like(b2s)
        b3s[:, 0, 0] = b1s[:, 1, 0] * b2s[:, 2, 0] - b1s[:, 2, 0] * b2s[:, 1, 0]
        b3s[:, 1, 0] = -(b1s[:, 0, 0] * b2s[:, 2, 0] - b1s[:, 2, 0] * b2s[:, 0, 0])
        b3s[:, 2, 0] = b1s[:, 0, 0] * b2s[:, 1, 0] - b1s[:, 1, 0] * b2s[:, 0, 0]
        Rs = torch.cat([b1s, b2s, b3s], dim=-1)
        return Rs

    def forward(
            self, pos: torch.FloatTensor,
            masks: torch.FloatTensor,
            inst_seg: torch.LongTensor,
            feats: {},
            conv_select_types=[0,0,0,0,0],
            loss_selection_dict={},
            loss_model: ComputingGraphLossModel_v5=None
        ):
        '''
        :param pos: position matrix with shape bz x N x 3
        :param masks:
        :param inst_seg:
        :param feats: dictionary containing other useful features
        :param conv_select_types:
        :param loss_selection_dict:
        :param loss_model:
        :return:
        '''

        pos = pos.float()
        bz, N = pos.size(0), pos.size(1)
        # nsmp = 256

        fps_idx = None

        pos = pos.contiguous()
        gt_l = torch.zeros((1,), dtype=torch.float32, device=pos.device)

        ''' GET features '''
        feat = []
        for k in feats:
            feat.append(feats[k])
        feat = torch.cat(feat, dim=-1) if len(feat) > 0 else None
        ''' GET features '''

        ''' GET feature embeddings '''
        if self.use_uniconv:
            # infeat = pos
            # print(feat.size(), pos.size())
            # infeat = torch.cat([pos, feat], dim=-1)
            x, global_x, pos = self.conv_uniunet(feat, pos=pos, return_global=True,
                                                 conv_select_types=conv_select_types)
        elif self.use_dgcnn:
            # infeat = pos
            infeat = torch.cat([pos, feat], dim=-1)
            #
            x, type_per_point, normal_per_point = self.conv_uniunet(xyz=infeat, normal=pos, inds=None,
                                                                    postprocess=False)
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

        ''' GET features & position for clustering seed points '''
        nseeds = self.nseeds

        losses = []
        sims = []

        if self.use_mixstyle:
            x = self.mixstyle(x)

        # gt_l += est

        pred_l = x

        if self.add_intermediat_loss: # or self.args.with_normalcst_loss or self.args.with_primpred_loss or self.args.with_parampred_loss:
            n_samples_inter_loss = self.args.n_samples_inter_loss # intermediate loss
            if n_samples_inter_loss < self.npoints:
                fps_idx = farthest_point_sampling(pos, n_sampling=n_samples_inter_loss)
            else:
                fps_idx = None

        # self.add_intermediat_loss = False
        ''' GET intermediate loss '''
        if self.add_intermediat_loss:
            matrices = {k: feats[k] for k in feats}
            masks = masks.float()
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

            #### ADD intermediate supervisions ####
            if len(loss_selection_dict) > 0:
                intermediate_loss = loss_model(
                    pos=pos_inter, x=pred_l_inter, label=masks_inter,
                    matrices=matrices,
                    oper_dict=loss_selection_dict
                )
                gt_l += intermediate_loss
        ''' GET intermediate loss '''

        ''' CLASSIFICATION for segmentation results '''
        confpred = None
        if self.cls_backbone:
            statistics = {}
            statistics['x'] = x
            ''' Propose segmentations in a top-down manner '''
            segpred_ori = CorrFlowPredNet.apply_module_with_conv1d_bn(
                x, self.cls_layers
            )
            segpred_clamped = torch.clamp(segpred_ori, min=-20, max=20)
            segpred_not_transposed = torch.softmax(segpred_clamped, dim=-1)

            # 3-interpolate?
            # pos_pos_sub_dist = torch.sum(())
            # segpred = batched_index_select(segpred, indices=minn_idx, dim=1)
            segpred = segpred_not_transposed.transpose(1, 2).contiguous()
            ''' Propose segmentations in a top-down manner '''

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

            ''' RT prediction '''
            if self.args.with_rt_loss:
                bz, N = pos.size(0), pos.size(1)
                # bz x N x 9
                R_t = CorrFlowPredNet.apply_module_with_conv1d_bn(
                    x, self.rt_pred
                )
                Rs = R_t[:, :, :6]
                ts = R_t[:, :, 6:]
                Rs = Rs.contiguous().view(bz, N, 3, 2).contiguous()
                Rs = self.get_rotation_matrix(Rs)
                Rs = Rs.contiguous().view(bz, N, -1).contiguous()
                statistics['pred_Rs'] = Rs
                statistics['pred_ts'] = ts
            ''' RT prediction '''

            masks_dim = masks.size(-1)
            if masks_dim > segpred.size(1):
                segpred = torch.cat(
                    [segpred, torch.zeros((bz, masks_dim - segpred.size(1), N), dtype=torch.float32, device=pos.device)],
                    dim=1
                )
                if self.args.with_conf_loss:
                    confpred = torch.cat(
                        [confpred,
                         torch.zeros((bz, masks_dim - confpred.size(1)), dtype=torch.float32, device=pos.device)],
                        dim=-1
                    )
            # print(masks.size(), segpred.size(), confpred.size())
            assert masks_dim <= segpred.size(1)

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

            grp_combined = grp_combined.squeeze(-1)
            grp_combined = torch.clamp(grp_combined, min=-20, max=20)

            ''' GET score for each mask '''
            _, sampled_pts_for_mask_idx = torch.topk(
                input=grp_combined.transpose(1, 2).contiguous(), dim=2, k=self.num_points_sampled_for_seed)
            # bz x nseeds x num_points_sampled_for_seed x k
            point_feat_expand = batched_index_select(values=x, indices=sampled_pts_for_mask_idx)
            point_feat_expand = point_feat_expand.contiguous().transpose(1, 2).contiguous()

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
            segpred = segpred * confpred.unsqueeze(-1)
            # momask_pred = momask_pred * confpred.unsqueeze(-1)
            # segpred = segpred + gt_logits_confpred.unsqueeze(-1)
            # segpred = torch.softmax(segpred, dim=1) # softmax for probability logits
            segpred = segpred / (torch.sum(segpred, dim=1, keepdim=True) + 1e-8)

            gt_l += gt_score_matching_loss # + pos_pp_loss + neg_pp_loss

            statistics = {"sim": sims}

        return segpred, gt_l, confpred, statistics, fps_idx

    def get_gt_conf(self, momask):
        # bz x nmask x N?
        momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf
