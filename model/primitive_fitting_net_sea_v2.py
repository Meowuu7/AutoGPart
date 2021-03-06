import torch
import torch.nn as nn
# from .pnmem_cell import DynamicRNN
from .utils import farthest_point_sampling, batched_index_select, get_knn_idx
from .model_util import construct_conv1d_modules, construct_conv_modules, CorrFlowPredNet
import torch.nn.functional as F
# from .loss_model import ComputingGraphLossModel
# from .loss_model_v2 import ComputingGraphLossModel as ComputingGraphLossModel_v2
# from .loss_model_v3 import ComputingGraphLossModel as ComputingGraphLossModel_v3
# from model.utils import iou, mean_shift, labels_to_one_hot_labels
# from torch.autograd import Variable

from .UNet_universal import UNet as UiUNet
from scipy.optimize import linear_sum_assignment
# from .point_convolution_universal import LocalConvNet, EdgeConv
from .Poinetnet2 import PointnetPP
from .DGCNN import PrimitiveNet

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

        self.stage = int(self.args.stage)

        self.num_points_sampled_for_seed = self.args.num_points_sampled_for_seed
        self.cls_backbone = self.args.cls_backbone
        self.with_normal = self.args.with_normal
        radius = self.args.radius.split(",")
        self.radius = [float(rr) for rr in radius]
        if not self.with_normal:
            self.in_feat_dim = self.in_feat_dim - 3 # exclude the normal feature vector
        ''' SET arguments '''

        ''' 6-layer feature extraction backbone '''
        if self.use_dgcnn:
            args.dgcnn_out_dim = 128
            map_feat_dim = 128
            feat_dims[-1] = 1024
            args.dgcnn_in_feat_dim = 6
            print("Using DGCNN")
            self.conv_uniunet = PrimitiveNet(args)

            args.dgcnn_in_feat_dim = args.intermediat_feat_pred_dim + 6
            # feature convolutional DGCNN
            if "further_pred" in args and args.further_pred:
                self.conv_uniunet_further = PrimitiveNet(args)
        elif self.use_spfn:
            self.conv_uniunet = PointnetPP(in_feat_dim=self.in_feat_dim + 3)
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
        # if self.add_intermediat_loss:
        #     if self.use_normal_loss:
        #         use_loss_model = ComputingGraphLossModel_v2
        #     else:
        #         use_loss_model = ComputingGraphLossModel
        #     self.intermediate_loss = use_loss_model(
        #         pos_dim=3, pca_feat_dim=64, in_feat_dim=map_feat_dim, pp_sim_k=16, r=0.03
        #     )
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

    def forward(
            self, pos: torch.FloatTensor,
            masks: torch.FloatTensor,
            inst_seg: torch.LongTensor,
            feats: {},
            conv_select_types=[0,0,0,0,0],
            loss_selection_dict={},
            loss_model=None # : ComputingGraphLossModel_v2=None
        ):

        # point_level_contrast, point_part_level_contrast, part_part_level_contrast = contrast_selection
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

        statistics = {}

        ''' GET features '''
        feat = []
        for k in feats:
            feat.append(feats[k])
        feat = torch.cat(feat, dim=-1) if len(feat) > 0 else None
        ''' GET features '''

        ''' GET feature embeddings '''
        if self.use_dgcnn:
            infeat = torch.cat([pos, feats["normals"]], dim=-1)
            x, type_per_point, normal_per_point = self.conv_uniunet(xyz=infeat, normal=feats["normals"], inds=None, postprocess=False)
            statistics['type_per_point'] = type_per_point
            statistics['normal_per_point'] = normal_per_point
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

        # losses = []
        sims = []
        gt_l = torch.zeros((1, ), dtype=torch.float32, device=pos.device)

        pred_l = x

        if self.add_intermediat_loss:
            # or self.args.with_normalcst_loss or self.args.with_primpred_loss or self.args.with_parampred_loss:
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

            # GET intermediate losses based on per-point embeddings
            intermediate_loss, predicted_feat = loss_model(
                pos=pos_inter, x=pred_l_inter, label=masks_inter,
                matrices=matrices,
                oper_dict=loss_selection_dict,
                return_predicted_feat=True
            )
            # predicted features
            statistics['predicted_feat'] = predicted_feat
            gt_l += intermediate_loss

            if self.stage == 2 and len(loss_selection_dict) > 0:
                predicted_feat = predicted_feat.detach()

            if len(loss_selection_dict) > 0 and self.stage == 2:
                # loss selection
                # further feature convolution
                infeat_further = torch.cat([pos, feats["normals"], predicted_feat], dim=-1)
                # normal vectors, input feature vectors;
                x, type_per_point, normal_per_point = self.conv_uniunet_further(xyz=infeat_further, normal=feats["normals"], inds=None,
                                                                        postprocess=False)

        ''' GET intermediate loss '''

        confpred = None
        if self.cls_backbone:

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

            #### EXCLUDE other losses ####
            # if self.args.with_primpred_loss or self.args.with_normalcst_loss or self.args.with_parampred_loss:
            #     if fps_idx is not None:
            #         x_sub = x.contiguous().view(bz * N, -1)[fps_idx, :].contiguous().view(bz, n_samples_inter_loss, -1).contiguous()
            #     else:
            #         x_sub = x
            #
            # ''' per-point primitive type prediction '''
            # if self.args.with_primpred_loss:
            #     prmpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
            #         x_sub, self.prm_pred_layers
            #     )
            #     # prmpred = torch.softmax(prmpred, dim=-1)
            #     prmpred = torch.clamp(prmpred, min=-20, max=20)
            #     prmpred = prmpred.transpose(1, 2).contiguous()
            #
            #     statistics["prmpred"] = prmpred
            # ''' per-point primitive type prediction '''
            #
            # ''' normal prediction '''
            # if self.args.with_normalcst_loss:
            #     normalpred = CorrFlowPredNet.apply_module_with_conv1d_bn(
            #         x_sub, self.normal_pred_layers
            #     )
            #     normalpred = torch.clamp(normalpred, min=-1e9, max=1e9)
            #     normalpred = torch.div(
            #         normalpred, torch.norm(normalpred, dim=-1, p=2, keepdim=True)
            #     )
            #     statistics["normalpred"] = normalpred
            # ''' normal prediction '''
            #
            # ''' PARAMETER prediction '''
            # if self.args.with_parampred_loss:
            #     parampred = CorrFlowPredNet.apply_module_with_conv1d_bn(
            #         x_sub, self.param_pred_layers
            #     )
            #     statistics["parampred"] = parampred
            # ''' PARAMETER prediction '''
            #### EXCLUDE other losses ####

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
            ''' GET point-point similarity loss '''

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
