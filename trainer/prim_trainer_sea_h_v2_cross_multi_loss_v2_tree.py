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
import horovod.torch as hvd
import torch.multiprocessing as mp
from filelock import FileLock
import time
import numpy as np
from model.utils import batched_index_select, calculate_acc
from .trainer_utils import MultiMultinomialDistribution
import logging
from model.loss_model_v5 import ComputingGraphLossModel
from .trainer_utils import get_masks_for_seg_labels, compute_param_loss, DistributionTreeNode, DistributionTreeNodeV2, DistributionTreeNodeArch
from model.constants import *


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
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        # torch.cuda.manual_seed(42)

        print("setting number threads...")
        torch.set_num_threads(5)

        kwargs = {'num_workers': 5, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'

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
        self.test_performance = self.args.test_performance

        self.base_loss_dicts = []

        if self.test_performance:
            self.nn_inter_loss = self.args.nn_inter_loss + len(self.base_loss_dicts)
        else:
            self.nn_inter_loss = self.args.nn_inter_loss
        self.nn_base_inter_loss = len(self.base_loss_dicts)

        if hvd.rank() == 0:
            print(f"Base loss dicts: {self.base_loss_dicts}")

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

        if len(self.resume) != 0:
            print(f"Loading weights from {self.resume}")
            ori_dict = torch.load(self.resume, map_location='cpu')
            part_dict = dict()
            model_dict = self.model.state_dict()
            for k in ori_dict:
                # if "feat_conv_net_trans_pred" not in k:
                if k in model_dict:
                    v = ori_dict[k]
                    part_dict[k] = v
            model_dict.update(part_dict)
            self.model.load_state_dict(model_dict)
            # self.model.load_state_dict(torch.load(resume, map_location='cpu'))

        self.model.cuda()
        # self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
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

        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if self.use_ansi:
                ansi_train_prim_types = self.args.ansi_train_prim_types.split(",")
                self.ansi_train_prim_types = [int(tpt) for tpt in ansi_train_prim_types]

                ansi_val_prim_types = self.args.ansi_val_prim_types.split(",")
                self.ansi_val_prim_types = [int(tpt) for tpt in ansi_val_prim_types]

                if self.test_performance:
                    self.ansi_train_prim_types = self.ansi_train_prim_types + self.ansi_val_prim_types

                ansi_test_prim_types = self.args.ansi_test_prim_types.split(",")
                self.ansi_test_prim_types = [int(tpt) for tpt in ansi_test_prim_types]

                self.split_train_test = self.args.split_train_test
                if self.args.test_performance:
                    self.split_train_test = True

                self.train_set = ANSIDataset(
                    root=self.dataset_root, filename=self.train_dataset, opt=self.args, csv_path="train_models.csv",
                    skip=1, fold=1,
                    prim_types=self.ansi_train_prim_types, split_type=self.split_type, split_train_test=self.split_train_test,
                    train_test_split="train", noisy=False,
                    first_n=-1, fixed_order=False
                )

                self.val_set = ANSIDataset(
                    root=self.dataset_root, filename=self.val_dataset, opt=self.args, csv_path="test_models.csv",
                    skip=1, fold=1,
                    prim_types=self.ansi_val_prim_types, split_type=self.split_type, split_train_test=self.split_train_test,
                    train_test_split="val", noisy=False,
                    first_n=-1, fixed_order=False
                )
            else:
                self.train_set = ABCDataset(
                    root=self.dataset_root, filename=self.train_dataset, opt=self.args, skip=1, fold=1,
                    prim_types=self.train_prim_types, split_type=self.split_type,
                    split_train_test=self.split_train_test, train_test_split="train"
                )
                self.test_set = ABCDataset(
                    root=self.dataset_root, filename=self.test_dataset, opt=self.args, skip=1, fold=1,
                    prim_types=self.test_prim_types, split_type=self.split_type, split_train_test=self.split_train_test,
                    train_test_split="test"
                )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set, num_replicas=hvd.size(), rank=hvd.rank())

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_set, num_replicas=hvd.size(), rank=hvd.rank())

        self.train_loader = data.DataLoader(
            self.train_set, batch_size=self.batch_size,
            sampler=self.train_sampler, **kwargs)

        self.val_loader = data.DataLoader(
            self.val_set, batch_size=self.batch_size,
            sampler=self.val_sampler, **kwargs)
        #### SET datasets & data-loaders & data-samplers ####

        ''' SET optimizer for the model '''
        lr_scaler = hvd.size()
        self.lr_scaler = lr_scaler
        if hvd.nccl_built():
            lr_scaler = hvd.local_size()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.init_lr * lr_scaler,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.model.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        ''' SET optimizer for the model '''

        ''' SET working dirs '''
        self.model_dir = "task_{}_nn_inter_loss_{}_inst_part_seg_mixing_type_{}_init_lr_{}_bsz_{}_drop_50_lr_schedule_projection_more_bn_resume_{}".format(
            self.args.task,
            str(self.nn_inter_loss),
            self.landmark_type,
            str(self.init_lr),
            str(batch_size),
            str(True if len(resume) > 0 else False)
        )

        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if not os.path.exists("./prm_cache"):
                os.mkdir("./prm_cache")
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir))
        self.model_dir = "./prm_cache/" + self.model_dir
        ''' SET working dirs '''

        ''' SET loss model '''
        self.loss_model_save_path = os.path.join(self.model_dir, "loss_model")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if not os.path.exists(self.loss_model_save_path):
                os.mkdir(self.loss_model_save_path)
        #### SET loss model version 5 ####
        self.loss_model = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                  lr_scaler=lr_scaler, init_lr=float(args.init_lr),
                                                  weight_decay=float(args.weight_decay),
                                                  loss_model_save_path=self.loss_model_save_path,
                                                  in_rep_dim=self.map_feat_dim if not args.use_spfn else 128,
                                                  nn_inter_loss=self.nn_inter_loss,
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
        # DISTRIBUTE self.head_optimizer
        self.head_optimizer = hvd.DistributedOptimizer(
            cur_optimizer,
            named_parameters=self.loss_model.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        #### BROADCAST the model's state_dict and the optimizer's state_dict ####
        hvd.broadcast_optimizer_state(self.head_optimizer, root_rank=0)
        hvd.broadcast_parameters(self.loss_model.state_dict(), root_rank=0)
        ''' SET optimizer for loss model '''

        ''' SET related sampling distributions '''
        self.nn_grp_opers = args.nn_grp_opers
        self.nn_binary_opers = args.nn_binary_opers
        self.nn_unary_opers = args.nn_unary_opers
        self.nn_in_feats = args.nn_in_feats
        ''' SET related sampling distributions '''

        ''' SET sampling tree '''
        if self.args.v2_tree:
            sampling_tree = DistributionTreeNodeV2
        else:
            sampling_tree = DistributionTreeNode

        self.best_val_acc = -9999.0

        if not args.debug and not args.pure_test:
            if not self.args.debug_arch:
                self.sampling_tree_rt = sampling_tree(cur_depth=0, nn_grp_opers=self.nn_grp_opers,
                                                      nn_binary_opers=self.nn_binary_opers, nn_unary_opers=self.nn_unary_opers,
                                                      nn_in_feat=self.nn_in_feats, args=args,
                                                      # forbid_dict=cur_forbid_dict, preference_dict=cur_preference_dict
                                                      )
            self.arch_dist = DistributionTreeNodeArch(cur_depth=0, tot_layers=3, nn_conv_modules=4, device=None,
                                                      args=None)
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
            # baseline_dict = self.get_baseline_intermediate_representation_generation()
            # for i in range(k):
            #     res.append(baseline_dict)
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

    def print_dist_params_to_file_arch(self):
        self.arch_dist.collect_params_and_save(os.path.join(self.model_dir, "dist_params_arch.npy"))

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
        # momask = momask.transpose(1, 2)
        # print("momask.size() =", momask.size(), "corr.size() =", corr.size())
        # bz x N x nmask
        momask_matrix = torch.sum(momask[:, :, None, :] * momask[:, None, :, :], dim=-1)
        # print("corr.type =", type(corr), "momask_matrix.type =", type(momask_matrix))
        loss = F.binary_cross_entropy_with_logits(input=corr, target=momask_matrix.float())
        loss = loss.mean()
        return loss

    def get_gt_conf(self, momask):
        # bz x nmask x N?
        #
        momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf

    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def _train_one_epoch(
            self, epoch,
            conv_select_types=[0, 0, 0, 0, 0],
            loss_selection_dict={}
        ):
        self.model.train()

        loss_list = []
        loss_nn = []
        train_bar = tqdm(self.train_loader)
        self.train_sampler.set_epoch(epoch)
        iouvalue = []

        avg_recall = []

        gt_loss = []

        for batch_dicts in train_bar:

            if self.use_ansi:
                batch_pos = batch_dicts['P']
                batch_normals = batch_dicts['normal_gt']
                batch_inst_seg = batch_dicts['I_gt']
                T_gt = batch_dicts['T_gt']
                batch_primitives = batched_index_select(values=T_gt, indices=batch_inst_seg, dim=1)
                # batch_primitives = batched_index_select(values=T_gt, indices=clean_batch_inst_seg, dim=1)
                # batch_primitives[batch_inst_seg == -1] = -1
            else:
                batch_pos = batch_dicts['gt_pc']
                batch_normals = batch_dicts['gt_normal']
                batch_inst_seg = batch_dicts['I_gt_clean']
                batch_primitives = batch_dicts['T_gt']
                batch_parameters = batch_dicts['T_param']

            if batch_pos.size(0) == 1:
                continue

            batch_pos = batch_pos.float().cuda()
            batch_normals = batch_normals.float().cuda()
            batch_inst_seg = batch_inst_seg.long().cuda()
            # batch_primitives = batch_primitives.long().cuda()
            if not self.use_ansi:
                batch_parameters = batch_parameters.float().cuda()

            batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
            bz, N = batch_pos.size(0), batch_pos.size(1)
            # print(batch_momasks.size(), self.n_max_instances)
            # assert batch_momasks.size(0) == bz and batch_momasks.size(2) == self.n_max_instances + 1 and batch_momasks.size(1) == N

            feats = {'normals': batch_normals}
            rt_values = self.model(
                pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                conv_select_types=conv_select_types,
                loss_selection_dict=loss_selection_dict,
                loss_model=self.loss_model
            )

            # separate losses for each mid-level prediction losses
            # losses = []
            seg_pred, gt_l, pred_conf, losses, fps_idx = rt_values

            if "iou" in losses:
                iou_value = losses["iou"]
                cur_avg_recall = 0.0
                seg_loss = None
            else:
                momasks_sub = batch_momasks
                # # momasks_sub size = bz x nsmp x nmasks
                batch_momasks, batch_conf = self.get_gt_conf(momasks_sub)

                batch_momasks = batch_momasks

                # try:
                iou_value, gt_conf, cur_avg_recall = iou(seg_pred, batch_momasks, batch_conf)
                iou_value = iou_value.mean()

                if self.args.with_conf_loss:
                    if batch_conf.size(1) < pred_conf.size(1):
                        tmp = pred_conf.clone()
                        bc = batch_conf.size(1)
                        tmp[:, :bc] = batch_conf
                        tmp[:, bc:] = 0.0
                        batch_conf = tmp

                    log_sigmoid_pred_conf = F.logsigmoid(pred_conf)
                    seg_loss = batch_conf * (-log_sigmoid_pred_conf) + \
                               (1. - batch_conf) * (-torch.log(torch.clamp(1. - torch.sigmoid(pred_conf), min=1e-10)))
                else:
                    seg_loss = None

            avg_recall.append(cur_avg_recall * bz)

            neg_iou_loss = -1.0 * iou_value

            loss = neg_iou_loss

            if self.args.add_intermediat_loss:
                loss += 0.1 * gt_l.mean()

            # loss = neg_iou_loss
            if seg_loss is not None and self.args.with_conf_loss:
                loss += seg_loss.mean()

            gt_loss.append(gt_l.detach().cpu().item() * bz)

            # n_sampling = self.args.n_samples_inter_loss

            self.head_optimizer.zero_grad()
            self.optimizer.zero_grad()
            # self.loss_model.head_optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            # self.loss_model.head_optimizer.step()
            self.head_optimizer.step()

            loss_list += [loss.detach().cpu().item() * bz]
            loss_nn.append(bz)
            try:
                iouvalue.append(-neg_iou_loss.detach().cpu().item() * bz)
            except:
                iouvalue.append(-neg_iou_loss * bz)

            train_bar.set_description(
                # 'Train Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Losses:{} Iou:{:.3f} AvgRecall:{:.3f} Record Iou:{}'.format(
                'Train Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Iou:{:.3f} AvgRecall:{:.3f}'.format(
                    epoch + 1, 200, float(sum(loss_list) / sum(loss_nn)),
                    float(sum(gt_loss)) / float(sum(loss_nn)),
                    # str(losses),
                    float(sum(iouvalue) / sum(loss_nn)),
                    float(sum(avg_recall) / sum(loss_nn)),
                    # str(thr_to_recall)
                ))

        avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
        avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
        avg_iou = float(sum(iouvalue)) / float(sum(loss_nn))
        avg_recall = float(sum(avg_recall)) / float(sum(loss_nn))
        avg_loss = self.metric_average(avg_loss, 'tr_loss')
        avg_gt_loss = self.metric_average(avg_gt_loss, 'tr_gt_loss')
        avg_iou = self.metric_average(avg_iou, 'tr_iou')
        avg_recall = self.metric_average(avg_recall, 'tr_avg_recall')

        if hvd.rank() == 0:
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("Train Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall:{:.3f}".format(
                    epoch + 1, avg_loss, avg_gt_loss,
                    avg_iou,
                    avg_recall
                ) + "\n")
                wf.close()

            logging.info("Train Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall:{:.3f}".format(
                epoch + 1, avg_loss, avg_gt_loss,
                avg_iou,
                avg_recall
            ))
        return avg_iou, avg_recall

    def _test(
        self, epoch, desc="val",
        conv_select_types=[0, 0, 0, 0, 0],
        loss_selection_dict=[]
    ):

        self.model.eval()

        with torch.no_grad():

            if desc == "val":
                cur_loader = self.val_loader
            elif desc == "test":
                cur_loader = self.test_loader
            else:
                raise ValueError(f"Unrecognized desc: {desc}")

            # cur_loader = self.test_loader

            loss_list = []
            loss_nn = []
            test_bar = tqdm(cur_loader)
            iouvalue = []

            gt_loss = []

            avg_recall = []

            for batch_dicts in test_bar:

                if self.use_ansi:
                    batch_pos = batch_dicts['P']
                    batch_normals = batch_dicts['normal_gt']
                    batch_inst_seg = batch_dicts['I_gt']
                    # T_gt = batch_dicts['T_gt']
                    # T_gt.size = bz x n_max_instance
                    # batch_primitives = batched_index_select(values=T_gt, indices=batch_inst_seg, dim=1)
                else:
                    batch_pos = batch_dicts['gt_pc']
                    batch_normals = batch_dicts['gt_normal']
                    batch_inst_seg = batch_dicts['I_gt_clean']
                    # batch_primitives = batch_dicts['T_gt']
                    # batch_parameters = batch_dicts['T_param']

                if batch_pos.size(0) == 1:
                    continue

                batch_pos = batch_pos.float().cuda()
                batch_normals = batch_normals.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                # batch_primitives = batch_primitives.long().cuda()
                # batch_parameters = batch_parameters.float().cuda()

                batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)

                bz, N = batch_pos.size(0), batch_pos.size(1)
                feats = {'normals': batch_normals}
                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model
                )

                # # separate losses for each mid-level prediction losses
                # losses = []
                seg_pred, gt_l, pred_conf, losses, fps_idx = rt_values

                if "iou" in losses:
                    iou_value = losses["iou"]
                    cur_avg_recall = 0.0
                    seg_loss = None
                else:
                    # nsmp = 256
                    # momasks_sub = batch_momasks.contiguous().view(bz * N, -1)[fps_idx, :].view(bz, nsmp, -1)
                    # #### For eal iou calculation ####
                    momasks_sub = batch_momasks
                    # momasks_sub size = bz x nsmp x nmasks
                    batch_momasks, batch_conf = self.get_gt_conf(momasks_sub)
                    batch_momasks = batch_momasks
                    #
                    try:
                        iou_value, gt_conf, cur_avg_recall = iou(seg_pred, batch_momasks, batch_conf)
                    except:
                        print(seg_pred)
                        continue
                    # print(iouvalue)
                    iou_value = iou_value.mean()

                    if self.args.with_conf_loss:
                        if batch_conf.size(1) < pred_conf.size(1):
                            tmp = pred_conf.clone()
                            bc = batch_conf.size(1)
                            tmp[:, :bc] = batch_conf
                            tmp[:, bc:] = 0.0
                            batch_conf = tmp

                        log_sigmoid_pred_conf = F.logsigmoid(pred_conf)
                        seg_loss = batch_conf * (-log_sigmoid_pred_conf) + \
                                   (1. - batch_conf) * (
                                       -torch.log(torch.clamp(1. - torch.sigmoid(pred_conf), min=1e-10)))
                    else:
                        seg_loss = None

                avg_recall.append(cur_avg_recall * bz)

                neg_iou_loss = -1.0 * iou_value

                loss = gt_l + neg_iou_loss

                if seg_loss is not None and self.args.with_conf_loss:
                    loss += seg_loss.mean()

                loss_list += [loss.detach().cpu().item() * bz]
                loss_nn.append(bz)
                try:
                    iouvalue.append(-neg_iou_loss.detach().cpu().item() * bz)
                except:
                    iouvalue.append(-neg_iou_loss * bz)

                test_bar.set_description(
                    # 'Test_{} Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Losses:{} Iou:{:.3f} AvgRecall:{:.3f} Recode Iou: {} '.format(
                    'Test_{} Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Iou:{:.3f} AvgRecall:{:.3f}'.format(
                        desc,
                        epoch + 1, 200, float(sum(loss_list) / sum(loss_nn)),
                        float(sum(gt_loss)) / float(sum(loss_nn)),
                        # str(losses),
                        float(sum(iouvalue) / sum(loss_nn)),
                        float(sum(avg_recall) / sum(loss_nn)),
                        # str(thr_to_recall)
                    ))

            avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
            avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
            avg_iou = float(sum(iouvalue)) / float(sum(loss_nn))
            avg_recall = float(sum(avg_recall)) / float(sum(loss_nn))
            avg_loss = self.metric_average(avg_loss, 'loss')
            avg_gt_loss = self.metric_average(avg_gt_loss, 'gt_loss')
            avg_iou = self.metric_average(avg_iou, 'iou')
            avg_recall = self.metric_average(avg_recall, 'avg_recall')

            if hvd.rank() == 0:
                with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                    wf.write("Test_{} Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall: {:.3f}".format(
                        desc,
                        epoch + 1,
                        avg_loss,
                        avg_gt_loss,
                        avg_iou,
                        avg_recall
                    ) + "\n")

                logging.info("Test_{} Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall: {:.3f}".format(
                    desc,
                    epoch + 1,
                    avg_loss,
                    avg_gt_loss,
                    avg_iou,
                    avg_recall
                ))
            return avg_iou, avg_recall

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "checkpoint_current.pth".format(epoch)))

    def rein_train_search_loss(
            self, base_epoch,
            base_arch_select_type,  # a list of selected baseline architecture
            baseline,
        ):
        eps_training_epochs = 1
        n_models_per_eps = 4
        tot_eps = 1

        final_baseline = 0.0

        for i_eps in range(tot_eps):
            best_model_test_acc = 0.0
            best_model_idx = 0
            rewards = []
            sampled_loss_dicts = []
            for i_model in range(n_models_per_eps):

                # cur_selected_loss_dict = self.sample_intermediate_representation_generation()
                cur_selected_loss_dict_list = self.sample_intermediate_representation_generation_k_list(
                    k=self.nn_inter_loss - self.nn_base_inter_loss, baseline=False)
                if hvd.rank() == 0:
                    logging.info(f"cur_selected_loss_dict = {cur_selected_loss_dict_list}")
                # cur_selected_loss_tsr = self.from_dist_dict_to_dist_tsr(cur_selected_loss_dict)
                cur_selected_loss_tsr = self.form_dist_dict_list_to_dist_tsr(cur_selected_loss_dict_list)

                cur_selected_loss_tsr = hvd.broadcast(cur_selected_loss_tsr, root_rank=0)

                # cur_selected_loss_dict = self.from_dist_tsr_to_dist_dict(cur_selected_loss_tsr)
                cur_selected_loss_dict_list = self.from_dist_tsr_to_dist_dict_list(cur_selected_loss_tsr)
                cur_selected_loss_dict_list = self.base_loss_dicts + cur_selected_loss_dict_list

                if hvd.rank() == 0:
                    logging.info(f"Sampled loss dict: {cur_selected_loss_dict_list}")
                if i_eps >= 0:
                    sampled_loss_dicts.append(cur_selected_loss_dict_list)

                    ''' LOAD model '''
                    self.model.load_state_dict(
                        torch.load(os.path.join(self.model_dir, "REIN_init_saved_model.pth"), map_location='cpu'
                                   )
                    )
                    self.model.cuda()  # .to(self.device)
                    ''' LOAD model '''

                    ''' LOAD loss model '''
                    # no i_model parameter is passed to the function
                    self.loss_model.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                           init_lr=self.init_lr,
                                                                           weight_decay=self.weight_decay)
                    self.loss_model.cuda()
                    ''' LOAD loss model '''
                    best_test_acc = 0.0

                    for i_iter in range(eps_training_epochs):
                        train_acc, train_rigid_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                        )

                        val_acc, _ = self._test(
                            base_epoch + i_iter + 1, desc="val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list
                            # r=cur_model_sampled_r
                        )
                        best_test_acc = val_acc - train_acc

                    rewards.append(best_test_acc)

                    if hvd.rank() == 0:
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_dir, f"REIN_saved_model_{i_model}.pth"))
                        # i_model parameter is passed to the function for clearer saving
                        self.loss_model.save_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list, i_model)

                        if best_test_acc > best_model_test_acc:
                            best_model_test_acc = best_test_acc
                            best_model_idx = i_model

                        logging.info(
                            f"{i_model + 1}-th model in {i_eps + 1}-th search with base epoch {base_epoch}: reward = {best_test_acc}, selected loss dict list = {cur_selected_loss_dict_list}")

                        with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                            wf.write(
                                f"{i_model + 1}-th model in {i_eps + 1}-th search with base epoch {base_epoch}: reward = {best_test_acc}, selected loss dict list = {cur_selected_loss_dict_list}" + "\n")
                            wf.close()

            if hvd.rank() == 0:
                self.update_dist_params_by_sampled_oper_dists_list_accum(sampled_loss_dicts, rewards=rewards,
                                                                         baseline=baseline, lr=self.init_lr)
                # todo: print updated distribution parameters
                print("Distribution parameters after update: ")
                # self.print_dist_params()
                self.print_dist_params_to_file()
                # Load weights
                self.loss_model.load_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx],
                                                                       init_lr=self.init_lr,
                                                                       weight_decay=self.weight_decay,
                                                                       model_idx=best_model_idx)
                # Save weights
                self.loss_model.save_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx])

            ''' LOAD best model '''
            logging.info(f"Loading from best model idx = {best_model_idx}")
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, f"REIN_saved_model_{best_model_idx}.pth"),
                           map_location="cpu"
                           ))
            self.model.cuda()
            ''' LOAD best model '''

            ''' SAVE as init_model '''
            if hvd.rank() == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))
            # get baseline
            best_test_acc = 0.0
            base_epoch += eps_training_epochs

            ''' GET baseline loss dict '''
            # baseline_loss_dict = self.get_baseline_intermediate_representation_generation()
            baseline_loss_dict = self.sample_intermediate_representation_generation_k_list(k=1, baseline=True)

            ''' BROADCAST baseline loss dict '''
            baseline_loss_tsr = self.form_dist_dict_list_to_dist_tsr(baseline_loss_dict)
            baseline_loss_tsr = hvd.broadcast(baseline_loss_tsr, root_rank=0)
            baseline_loss_dict = self.from_dist_tsr_to_dist_dict_list(baseline_loss_tsr)
            baseline_loss_dict = self.base_loss_dicts + baseline_loss_dict

            ''' LOAD loss model by baseline loss dict '''
            self.loss_model.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                   weight_decay=self.weight_decay)
            self.loss_model.cuda()

            for i_iter in range(eps_training_epochs):
                train_acc, train_rigid_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict
                    # r=baseline_r
                )

                val_acc, _ = self._test(
                    base_epoch + i_iter + 1, desc="val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict
                )
                # val_acc = test_acc
                best_test_acc = val_acc - train_acc
                # best_test_acc = val_acc - train_acc

            baseline = best_test_acc
            if hvd.rank() == 0:
                logging.info(f"Current baseline = {baseline}")
                logging.info(f"Baseline model dictionary selection: {baseline_loss_dict}")
                with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                    wf.write(f"Current baseline = {baseline}" + "\n")
                    wf.write(f"Baseline model dictionary selection = {baseline_loss_dict}" + "\n")
                    wf.close()
            final_baseline = baseline
            final_baseline_loss_dict = baseline_loss_dict

        return final_baseline, final_baseline_loss_dict

    def train_all(self):
        # eps_training_epochs = 10
        eps_training_epochs = 1
        # test for initial skip connection probability
        # eps_training_epochs = 400
        n_models_per_eps = 4
        tot_eps = 30
        ''' SAVE current model weights as initial weights '''
        if hvd.rank() == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))

        if not self.test_performance:
            baseline_loss_dict = []

        best_val_acc = 0.0
        best_test_acc = 0.0

        baseline_value = torch.tensor([0, 0, 0], dtype=torch.long)

        not_improved_num_epochs = 0
        if self.test_performance:

            baseline_loss_dict = []
            ''' LOAD model '''

            ''' LOAD loss model (head & optimizer) '''
            # LOAD heads for features prediction
            self.loss_model.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                   weight_decay=self.weight_decay)
            self.loss_model.cuda()
            ''' LOAD loss model (head & optimizer) '''
            eps_training_epochs = 400

        best_test_val_acc = 0.0
        if self.test_performance or not self.test_performance:
            if self.test_performance:
                self.args.add_intermediat_loss = False
            for i_iter in range(eps_training_epochs):
                train_acc, train_rigid_loss = self._train_one_epoch(
                    i_iter + 1,
                    conv_select_types=baseline_value.tolist(),
                    loss_selection_dict=baseline_loss_dict
                )

                # if not self.test_performance:
                val_acc, _ = self._test(
                    i_iter + 1, desc="val",
                    conv_select_types=baseline_value.tolist(),
                    loss_selection_dict=baseline_loss_dict
                    # r=cur_model_sampled_r
                )

                best_test_acc = val_acc - train_acc

                if self.test_performance:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        not_improved_num_epochs = 0
                        if hvd.rank() == 0:
                            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_best_saved_model.pth"))
                    else:
                        not_improved_num_epochs += 1
                        if not_improved_num_epochs >= 20:
                            not_improved_num_epochs = 0
                            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                                wf.write(f"Adjusting learning rate by {0.7}.\n")
                                wf.close()
                            self.adjust_learning_rate_by_factor(0.7)

            print(f"Val acc = {best_val_acc}.")
            return

        baseline = best_test_acc

        base_epoch = 0
        if hvd.rank() == 0:
            logging.info(f"Baseline = {baseline}")

        each_search_epochs = self.sea_interval
        base_model_select_arch_types = baseline_value.tolist()

        for i_eps in range(tot_eps):

            ''' SUP search '''
            baseline, baseline_loss_dict = self.rein_train_search_loss(
                base_epoch=base_epoch, base_arch_select_type=base_model_select_arch_types, baseline=baseline
            )

            base_epoch += each_search_epochs
