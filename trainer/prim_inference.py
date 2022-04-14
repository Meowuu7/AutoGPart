import torch
import torch.nn as nn
from model.pointnetpp_segmodel_sea import PointNetPPInstSeg
from model.primitive_fitting_net_sea import PrimitiveFittingNet

# from datasets.Indoor3DSeg_dataset import Indoor3DSemSeg
from datasets.instseg_dataset import InstSegmentationDataset
from datasets.ABC_dataset import ABCDataset
from datasets.ANSI_dataset import ANSIDataset
from torch.nn import functional as F

from tqdm import tqdm
import os
from torch.utils import data
from model.utils import iou
# import horovod.torch as hvd
import torch.multiprocessing as mp
# from filelock import FileLock
import time
import numpy as np
from model.utils import batched_index_select, calculate_acc
from .trainer_utils import MultiMultinomialDistribution
import logging
from model.loss_model_v5 import ComputingGraphLossModel
from .trainer_utils import get_masks_for_seg_labels, compute_param_loss, DistributionTreeNode, DistributionTreeNodeV2
from model.utils import mean_shift
from model.loss_utils import compute_miou, npy
from model.abc_utils import compute_entropy, construction_affinity_matrix_normal
from model.loss_utils import compute_embedding_loss, compute_normal_loss, compute_nnl_loss


class TrainerPrimitiveFitting(nn.Module):
    def __init__(self, dataset_root, num_points=512, batch_size=32, num_epochs=200, cuda=None, dataparallel=False,
                 use_sgd=False, weight_decay_sgd=5e-4,
                 resume="", dp_ratio=0.5, args=None):
        super(TrainerPrimitiveFitting, self).__init__()
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        self.num_epochs = num_epochs
        if cuda is not None:
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        print("Before init")

        torch.cuda.set_device(args.gpu)

        print("setting number threads...")
        torch.set_num_threads(5)

        kwargs = {'num_workers': 5, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

        self.kwargs = kwargs

        lr_scaler = 1.0

        self.args = args
        self.batch_size = int(self.args.batch_size)
        self.num_epochs = int(self.args.epochs)
        self.dp_ratio = float(self.args.dp_ratio)
        self.landmark_type = args.landmark_type
        self.init_lr = float(self.args.init_lr)
        self.weight_decay = float(self.args.weight_decay)
        self.lr_decay_ratio = float(self.args.lr_decay_ratio)
        self.step_size = int(self.args.step_size)
        self.n_points = int(self.args.num_points)
        self.resume = self.args.resume
        feat_dims = self.args.feat_dims.split(",")
        self.feat_dims = [int(fd) for fd in feat_dims]
        self.use_ansi = self.args.use_ansi
        self.with_fitting_loss = self.args.with_fitting_loss
        self.n_max_instances = self.args.n_max_instances
        self.sea_interval = self.args.sea_interval

        self.nn_inter_loss = self.args.nn_inter_loss

        n_samples = self.args.n_samples.split(",")
        self.n_samples = [int(ns) for ns in n_samples]
        self.map_feat_dim = int(self.args.map_feat_dim)
        self.n_layers = int(self.args.n_layers)
        assert self.n_layers == len(
            self.n_samples), f"Expect the times of down-sampling equal to n_layers, got n_layers = {self.n_layers}, times of down-sampling = {len(self.n_samples)}."
        assert self.n_layers == len(
            self.feat_dims), f"Expect the number of feature dims equal to n_layers, got n_layers = {self.n_layers}, number of dims = {len(self.feat_dims)}."

        ''' GET model & loss selection parameters '''
        conv_select_types = self.args.conv_select_types.split(",")
        self.conv_select_types = [int(tt) for tt in conv_select_types]
        self.point_feat_selection = int(self.args.point_feat_selection)
        self.point_geo_feat_selection = int(self.args.point_geo_feat_selection)
        # print(self.args.contrast_selection)
        contrast_selection = self.args.contrast_selection.split(",")
        self.contrast_selection = [int(cs) for cs in contrast_selection]

        #### GET model ####
        self.model = PrimitiveFittingNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args
        )

        assert len(self.resume) != 0
        if len(self.resume) != 0:
            print(f"Loading weights from {self.resume}")
            ori_dict = torch.load(self.resume, map_location='cpu')
            part_dict = dict()
            model_dict = self.model.state_dict()
            for k in ori_dict:
                if k in model_dict:
                    v = ori_dict[k]
                    part_dict[k] = v
            model_dict.update(part_dict)
            self.model.load_state_dict(model_dict)

        self.model.cuda()
        #### GET model ####

        ### SET datasets & data-loaders & data-samplers ####
        self.dataset_root = self.args.dataset_root
        self.nmasks = int(self.args.nmasks)
        self.train_dataset = self.args.train_dataset
        self.val_dataset = self.args.val_dataset
        self.test_dataset = self.args.test_dataset
        self.split_type = self.args.split_type
        self.split_train_test = self.args.split_train_test
        train_prim_types = self.args.train_prim_types.split(",")
        self.train_prim_types = [int(tpt) for tpt in train_prim_types]
        # val_prim_types = self.args.val_prim_types.split(",")
        # self.val_prim_types = [int(tpt) for tpt in val_prim_types]
        test_prim_types = self.args.test_prim_types.split(",")
        self.test_prim_types = [int(tpt) for tpt in test_prim_types]

        ### SET pure test setting ####
        self.test_performance = self.args.test_performance

        ''' SET working dirs '''
        self.model_dir = "task_{}_nn_inter_loss_{}_inst_part_seg_mixing_type_{}_init_lr_{}_bsz_{}_drop_50_lr_schedule_projection_more_bn_resume_{}".format(
            self.args.task,
            str(self.nn_inter_loss),
            self.landmark_type,
            str(self.init_lr),
            str(batch_size),
            str(True if len(resume) > 0 else False)
        )

        # with FileLock(os.path.expanduser("~/.horovod_lock")):
        if not os.path.exists("./prm_cache"):
            os.mkdir("./prm_cache")
        if not os.path.exists(os.path.join("./prm_cache", self.model_dir)):
            os.mkdir(os.path.join("./prm_cache", self.model_dir))
        self.model_dir = "./prm_cache/" + self.model_dir
        ''' SET working dirs '''

        ''' SET loss model '''
        self.loss_model_save_path = os.path.join(self.model_dir, "loss_model")
        if not os.path.exists(self.loss_model_save_path):
            os.mkdir(self.loss_model_save_path)
        #### SET loss model version 5 ####
        self.loss_model = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                  lr_scaler=lr_scaler, init_lr=float(args.init_lr),
                                                  weight_decay=float(args.weight_decay),
                                                  loss_model_save_path=self.loss_model_save_path,
                                                  in_rep_dim=self.map_feat_dim if not args.use_spfn else 128,
                                                  nn_inter_loss=args.nn_inter_loss,
                                                  args=args
                                                  )
        self.loss_model.cuda()
        ''' SET loss model '''

        ''' SET optimizer for loss model '''
        cur_optimizer = torch.optim.Adam(
            self.loss_model.parameters(),
            lr=self.init_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)
        self.head_optimizer = cur_optimizer
        ''' SET optimizer for loss model '''

        ''' SET related sampling distributions '''
        # number of grouping operations, binary operations and unary operations
        self.nn_grp_opers = args.nn_grp_opers
        self.nn_binary_opers = args.nn_binary_opers
        self.nn_unary_opers = args.nn_unary_opers
        self.nn_in_feats = args.nn_in_feats
        ''' SET related sampling distributions '''

        if self.args.add_intermediat_loss:
            ''' SET sampling tree '''
            if self.args.v2_tree:
                sampling_tree = DistributionTreeNodeV2
            else:
                sampling_tree = DistributionTreeNode

            self.sampling_tree_rt = sampling_tree(cur_depth=0, nn_grp_opers=self.nn_grp_opers,
                                                  nn_binary_opers=self.nn_binary_opers, nn_unary_opers=self.nn_unary_opers,
                                                  nn_in_feat=self.nn_in_feats, args=args)
        ''' SET related sampling distributions '''

    def adjust_learning_rate_by_factor(self, scale_factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale_factor

    def from_dist_dict_to_dist_tsr(self, dist_dict):
        cur_oper_tsr = []
        cur_oper_tsr.append(dist_dict["uop"] if "uop" in dist_dict else -1)
        cur_oper_tsr.append(dist_dict["gop"] if "gop" in dist_dict else -1)
        cur_oper_tsr.append(dist_dict["bop"] if "bop" in dist_dict else -1)
        cur_oper_tsr.append(dist_dict["oper"] if "oper" in dist_dict else -1)
        if "chd" in dist_dict:
            lft_chd_tsr = self.from_dist_dict_to_dist_tsr(dist_dict["chd"])
            lft_chd_size = lft_chd_tsr.size(0)
            cur_oper_tsr += [lft_chd_size, 0]
            cur_oper_tsr = torch.cat([torch.tensor(cur_oper_tsr, dtype=torch.long).cuda(), lft_chd_tsr], dim=0)
        elif "lft_chd" in dist_dict:
            lft_chd_tsr = self.from_dist_dict_to_dist_tsr(dist_dict["lft_chd"])
            rgt_chd_tsr = self.from_dist_dict_to_dist_tsr(dist_dict["rgt_chd"])
            cur_oper_tsr += [lft_chd_tsr.size(0), rgt_chd_tsr.size(0)]
            cur_oper_tsr = torch.cat(
                [torch.tensor(cur_oper_tsr, dtype=torch.long).cuda(), lft_chd_tsr, rgt_chd_tsr], dim=0
            )
        else:
            cur_oper_tsr += [0, 0]
            cur_oper_tsr = torch.tensor(cur_oper_tsr, dtype=torch.long).cuda()

        return cur_oper_tsr

    def form_dist_dict_list_to_dist_tsr(self, dist_dict_list):
        oper_tsr_list = []
        maxx_sz = 0

        for i, dist_dict in enumerate(dist_dict_list):
            cur_tsr = self.from_dist_dict_to_dist_tsr(dist_dict)
            oper_tsr_list.append(cur_tsr)
            cur_tsr_sz = cur_tsr.size(0)
            if cur_tsr_sz > maxx_sz:
                maxx_sz = cur_tsr_sz

        maxx_sz = 200
        pad_oper_tsr_list = []
        for i, oper_tsr in enumerate(oper_tsr_list):
            # Pad value -2 for not filled-in values
            if oper_tsr.size(0) < maxx_sz:
                pad_oper_tsr = torch.cat(
                    [oper_tsr, torch.full((maxx_sz - oper_tsr.size(0),), fill_value=-2, dtype=torch.long).cuda()],
                    dim=0)
            else:
                pad_oper_tsr = oper_tsr
            pad_oper_tsr_list.append(pad_oper_tsr.unsqueeze(0))

        pad_oper_tsr = torch.cat(pad_oper_tsr_list, dim=0)

        return pad_oper_tsr

    def from_oper_list_to_oper_dict(self, oper_list):
        cur_opers = oper_list[:6]
        cur_oper_dict = {}
        if cur_opers[1] != -1:
            cur_oper_dict["gop"] = cur_opers[1]
        if cur_opers[0] != -1:
            cur_oper_dict["uop"] = cur_opers[0]

        if cur_opers[2] != -1:
            cur_oper_dict["bop"] = cur_opers[2]
        if cur_opers[3] != -1:
            cur_oper_dict["oper"] = cur_opers[3]
        if cur_opers[4] != 0 and cur_opers[5] == 0:
            chd_oper_list = oper_list[6: 6 + cur_opers[4]]
            chd_oper_dict = self.from_oper_list_to_oper_dict(chd_oper_list)
            cur_oper_dict["chd"] = chd_oper_dict
        if cur_opers[4] != 0 and cur_opers[5] != 0:
            lft_chd_oper_list = oper_list[6: 6 + cur_opers[4]]
            rgt_chd_oper_list = oper_list[6 + cur_opers[4]: 6 + cur_opers[4] + cur_opers[5]]
            lft_chd_oper_dict = self.from_oper_list_to_oper_dict(lft_chd_oper_list)
            rgt_chd_oper_dict = self.from_oper_list_to_oper_dict(rgt_chd_oper_list)
            cur_oper_dict["lft_chd"] = lft_chd_oper_dict
            cur_oper_dict["rgt_chd"] = rgt_chd_oper_dict
        return cur_oper_dict

    def from_dist_tsr_to_dist_dict(self, dist_tsr):
        # n_elements = dist_tsr.size(0)
        dist_list = dist_tsr.detach().tolist()
        dist_list = [int(dl) for dl in dist_list]
        clean_dist_list = [dl for dl in dist_list if dl != -2]

        dist_dict = self.from_oper_list_to_oper_dict(clean_dist_list)

        return dist_dict

    def from_dist_tsr_to_dist_dict_list(self, dist_tsr):
        n_samples = dist_tsr.size(0)
        dist_dict_list = []
        for i in range(n_samples):
            cur_dict = self.from_dist_tsr_to_dist_dict(dist_tsr[i])
            dist_dict_list.append(cur_dict)
        return dist_dict_list

    def sample_intermediate_representation_generation(self):
        ret_dict = self.sampling_tree_rt.sampling(cur_depth=0)
        return ret_dict

    def get_baseline_intermediate_representation_generation(self):
        ret_dict = self.sampling_tree_rt.baseline_sampling(cur_depth=0)
        return ret_dict

    ''' SAMPLE intermediate loss generation dictionaries '''
    def sample_intermediate_representation_generation_k_list(self, k=1, baseline=False):
        res = []
        if baseline:
            res = self.sampling_tree_rt.sample_basline_oper_dict_list()
        else:
            for j in range(k):
                sampled_loss_dict = self.sample_intermediate_representation_generation()
                res.append(sampled_loss_dict)
        return res

    def update_dist_params_by_sampled_oper_dists_list_accum(self, oper_dist_list, rewards, baseline, lr):
        n_samples = len(oper_dist_list)
        ''' ACCUMULATE operator selection & transformation selection distributions '''
        self.sampling_tree_rt.update_sampling_dists_by_sampled_dicts(oper_dist_list, rewards, baseline, lr=lr)

    def print_dist_params(self):
        print("=== Distribution parameters ===")
        self.sampling_tree_rt.print_params()

    def print_dist_params_to_file(self):
        if self.args.v2_tree:
            self.sampling_tree_rt.collect_params_and_save(os.path.join(self.model_dir, "dist_params.npy"))
        else:
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("=== Distribution parameters ===" + "\n")
                self.sampling_tree_rt.print_params_to_file(wf)
                wf.close()

    def get_loss_part_seg(self, corr: torch.FloatTensor, momask: torch.LongTensor):
        nmask = momask.size(1)
        momask_matrix = torch.sum(momask[:, :, None, :] * momask[:, None, :, :], dim=-1)
        # print("corr.type =", type(corr), "momask_matrix.type =", type(momask_matrix))
        loss = F.binary_cross_entropy_with_logits(input=corr, target=momask_matrix.float())
        loss = loss.mean()
        return loss

    def get_gt_conf(self, momask):
        momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf

    def _clustering_test(
        self, epoch, desc="val",
        conv_select_types=[0, 0, 0, 0, 0],
        loss_selection_dict=[],
        save_samples=False,
        sample_interval=20,
        inference_prim_types=[2,6]
    ):
        save_stats_path = os.path.join(self.model_dir, "inference_saved")
        inference_prim_types_str = [str(ttt) for ttt in inference_prim_types]

        self.args.inference = False
        self.test_set = ANSIDataset(
            root=self.dataset_root, filename=self.test_dataset, opt=self.args, csv_path="test_models.csv",
            skip=1, fold=1,
            prim_types=inference_prim_types, split_type=self.split_type, split_train_test=False,
            train_test_split="test", noisy=False,
            first_n=-1, fixed_order=False
        )

        self.test_loader = data.DataLoader(
            self.test_set, batch_size=1,
            shuffle=False, **self.kwargs)

        test_bar = tqdm(self.test_loader)

        self.model.eval()

        poses, gt_segs, pred_segs = [], [], []
        normals = []

        all_miou = []

        feat_losses = []
        normal_losses = []
        type_losses = []
        all_miou = []
        loss_nn = []

        if save_samples:
            poes, gt_segs, pred_segs, normals = [], [], [], []

        with torch.no_grad():

            for i_batch, batch_dicts in enumerate(test_bar):
                if save_samples and (not (i_batch % sample_interval == 0)):
                    continue

                if self.use_ansi:
                    batch_pos = batch_dicts['P']
                    batch_normals = batch_dicts['normal_gt']
                    batch_inst_seg = batch_dicts['I_gt']
                    T_gt = batch_dicts['T_gt']
                    batch_primitives = batched_index_select(values=T_gt, indices=batch_inst_seg, dim=1)
                else:
                    batch_pos = batch_dicts['gt_pc']
                    batch_normals = batch_dicts['gt_normal']
                    batch_inst_seg = batch_dicts['I_gt_clean']

                batch_pos = batch_pos.float().cuda()
                batch_normals = batch_normals.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                batch_primitives = batch_primitives.long().cuda()


                batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)

                # bz, N = batch_pos.size(0), batch_pos.size(1)
                feats = {'normals': batch_normals}
                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model
                )

                seg_pred, gt_l, pred_conf, statistics, fps_idx = rt_values
                # print("here 0!")

                feat_ent_weight = 1.70
                edge_ent_weight = 1.23
                edge_knn = 50
                normal_sigma = 0.1
                edge_topK = 12
                bandwidth = 0.85

                spec_embedding_list = []
                weight_ent = []

                # bz x N x k
                x = statistics['x']
                # x = x.detach().cpu()

                pred_type = statistics['type_per_point']
                normal_per_point = statistics['normal_per_point']
                feat_loss, pull_loss, push_loss, _, _ = compute_embedding_loss(x, batch_inst_seg)
                normal_loss = compute_normal_loss(normal_per_point, batch_normals)
                type_loss = compute_nnl_loss(pred_type, batch_primitives)

                feat_losses.append(feat_loss.detach().cpu().item() * 1)
                normal_losses.append(normal_loss.detach().cpu().item() * 1)
                type_losses.append(type_loss.detach().cpu().item() * 1)

                loss_nn.append(1)

                feat_ent = feat_ent_weight - float(npy(compute_entropy(x)))

                weight_ent.append(feat_ent)
                spec_embedding_list.append(x)

                affinity_matrix_normal = construction_affinity_matrix_normal(batch_pos, batch_normals,
                                                                             sigma=normal_sigma,
                                                                             knn=edge_knn)
                edge_topk = edge_topK
                e, v = torch.lobpcg(affinity_matrix_normal, k=edge_topk, niter=10)
                v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-16)
                edge_ent = edge_ent_weight - float(npy(compute_entropy(v)))

                weight_ent.append(edge_ent)
                spec_embedding_list.append(v)

                weighted_list = []
                for i in range(len(spec_embedding_list)):
                    weighted_list.append(spec_embedding_list[i] * weight_ent[i])

                spectral_embedding = torch.cat(weighted_list, dim=-1)

                spec_cluster_pred = mean_shift(spectral_embedding, bandwidth=bandwidth)

                miou = compute_miou(spec_cluster_pred, batch_inst_seg)

                all_miou.append(miou.detach().item())

                test_bar.set_description(
                    'Test_{} Epoch: [{}/{}] Iou:{:.3f} feat_loss:{:.3f} normal_loss:{:.3f} type_loss:{:.3f}'.format(
                        desc,
                        epoch + 1, 200, float(sum(all_miou) / len(all_miou)),
                        float(sum(feat_losses) / sum(loss_nn)),
                        float(sum(normal_losses)) / float(sum(loss_nn)),
                        # str(losses),
                        float(sum(type_losses) / sum(loss_nn)),
                    ))

            with open(os.path.join(self.model_dir, "test_logs.txt"), "a") as wf:
                prim_strr = ",".join(inference_prim_types_str)
                wf.write(f"{prim_strr}: " + 'Iou:{:.3f} feat_loss:{:.3f} normal_loss:{:.3f} type_loss:{:.3f}'.format(
                        float(sum(all_miou) / len(all_miou)),
                        float(sum(feat_losses) / sum(loss_nn)),
                        float(sum(normal_losses)) / float(sum(loss_nn)),
                        # str(losses),
                        float(sum(type_losses) / sum(loss_nn)),) + "\n")
                wf.close()
            return

    def _test(
        self, epoch, desc="val",
        conv_select_types=[0, 0, 0, 0, 0],
        loss_selection_dict=[],
        cur_ansi_test_prim_type=0
    ):

        save_stats_path = os.path.join(self.model_dir, "inference_saved")

        # self.args.inference =
        self.test_set = ANSIDataset(
            root=self.dataset_root, filename=self.test_dataset, opt=self.args, csv_path="test_models.csv",
            skip=1, fold=1,
            prim_types=cur_ansi_test_prim_type, split_type=self.split_type, split_train_test=False,
            train_test_split="test", noisy=False,
            first_n=-1, fixed_order=False
        )

        self.test_loader = data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False, **self.kwargs)

        test_bar = tqdm(self.test_loader)

        self.model.eval()

        poses, gt_segs, pred_segs = [], [], []
        normals = []
        all_miou = []
        nns = []

        with torch.no_grad():

            for batch_dicts in test_bar:

                if self.use_ansi:
                    batch_pos = batch_dicts['P']
                    batch_normals = batch_dicts['normal_gt']
                    batch_inst_seg = batch_dicts['I_gt']
                    T_gt = batch_dicts['T_gt']
                else:
                    batch_pos = batch_dicts['gt_pc']
                    batch_normals = batch_dicts['gt_normal']
                    batch_inst_seg = batch_dicts['I_gt_clean']

                # if batch_pos.size(0) == 1:
                #     continue

                batch_pos = batch_pos.float().cuda()
                batch_normals = batch_normals.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()

                batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)

                # bz, N = batch_pos.size(0), batch_pos.size(1)
                feats = {'normals': batch_normals}
                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model
                )

                seg_pred, gt_l, pred_conf, statistics, fps_idx = rt_values

                per_point_seg_pred = torch.argmax(seg_pred, dim=1)

                for j in range(batch_pos.size(0)):

                    miou = compute_miou(per_point_seg_pred[j].unsqueeze(0), batch_inst_seg[j].unsqueeze(0))

                    all_miou.append(miou.detach().item())

                nns.append(batch_pos.size(0))

                test_bar.set_description(
                    'Test_{} Epoch: [{}/{}] Iou:{:.3f}'.format(
                        desc,
                        epoch + 1, 200, float(sum(all_miou)) / float(sum(nns)),
                    ))

            return

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "checkpoint_current.pth".format(epoch)))

    def train_all(self):

        if not os.path.exists(os.path.join(self.model_dir, "inference_saved")):
            os.mkdir(os.path.join(self.model_dir, "inference_saved"))

        self.args.inference = False
        self.args.add_intermediat_loss = False

        # jj = 6
        for jj in [2,6]:
            self._clustering_test(0, "test", [0,0,0], loss_selection_dict=[], save_samples=False, inference_prim_types=[jj])
