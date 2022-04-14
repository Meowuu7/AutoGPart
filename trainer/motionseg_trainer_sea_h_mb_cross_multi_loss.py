import torch
import torch.nn as nn
# from model.pointnetpp_segmodel_sea import PointNetPPInstSeg
# from model.primitive_fitting_net_sea import PrimitiveFittingNet
# from model.pointnetpp_segmodel_cls_sea import InstSegNet
# from model.pointnetpp_segmodel_cls_sea_v2 import InstSegNet
from model.motion_segmodel_cls_sea_v2 import InstSegNet

# from datasets.Indoor3DSeg_dataset import Indoor3DSemSeg
# from datasets.instseg_dataset import InstSegmentationDataset
# from datasets.ABC_dataset import ABCDataset
# from datasets.ANSI_dataset import ANSIDataset
from datasets.motionseg_dataset import PartSegmentationMetaInfoDataset
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
# from model.utils import batched_index_select, calculate_acc
# from .trainer_utils import MultiMultinomialDistribution
import logging
from model.loss_model_v5 import ComputingGraphLossModel
from .trainer_utils import get_masks_for_seg_labels, compute_param_loss, DistributionTreeNode, DistributionTreeNodeV2, DistributionTreeNodeArch
# from datasets.partnet_dataset import PartNetInsSeg
# from model.loss_utils import get_one_hot
# from model.model_util import set_bn_not_training, set_grad_to_none

class TrainerInstSegmentation(nn.Module):
    def __init__(self, dataset_root, num_points=512, batch_size=32, num_epochs=200, cuda=None, dataparallel=False,
                 use_sgd=False, weight_decay_sgd=5e-4,
                 resume="", dp_ratio=0.5, args=None):
        super(TrainerInstSegmentation, self).__init__()
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        self.num_epochs = num_epochs
        if cuda is not None:
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        ''' SET some basic configurations '''
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

        if hvd.size() > 1:
            torch.set_num_threads(5)
            kwargs = {'num_workers': 5, 'pin_memory': True}
        else:
            kwargs = {}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        self.kwargs = kwargs
        ''' SET some basic configurations '''

        ''' SET arguments '''
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

        # number of added intermediate losses
        self.nn_inter_loss = self.args.nn_inter_loss # 1

        n_samples = self.args.n_samples.split(",")
        self.n_samples = [int(ns) for ns in n_samples]
        self.map_feat_dim = int(self.args.map_feat_dim)
        self.n_layers = int(self.args.n_layers)
        assert self.n_layers == len(
            self.n_samples), f"Expect the times of down-sampling equal to n_layers, got n_layers = {self.n_layers}, times of down-sampling = {len(self.n_samples)}."
        assert self.n_layers == len(
            self.feat_dims), f"Expect the number of feature dims equal to n_layers, got n_layers = {self.n_layers}, number of dims = {len(self.feat_dims)}."
        ''' SET arguments '''

        ''' GET model & loss selection parameters '''
        conv_select_types = self.args.conv_select_types.split(",")
        self.conv_select_types = [int(tt) for tt in conv_select_types]
        self.point_feat_selection = int(self.args.point_feat_selection)
        self.point_geo_feat_selection = int(self.args.point_geo_feat_selection)
        # print(self.args.contrast_selection)
        contrast_selection = self.args.contrast_selection.split(",")
        self.contrast_selection = [int(cs) for cs in contrast_selection]
        ''' GET model & loss selection parameters ---- history '''

        self.no_spectral = self.args.no_spectral

        ''' SET working dirs '''
        self.model_dir = "task_{}_debug_{}_npoints_{}_nn_inter_loss_{}_in_model_loss_model_{}_inst_part_seg_mixing_type_{}_init_lr_{}_bsz_{}_weight_decay_{}_more_bn_resume_{}".format(
            self.args.task,
            str(args.debug),
            str(self.n_points),
            str(self.nn_inter_loss),
            str(args.in_model_loss_model),
            self.landmark_type,
            str(self.init_lr),
            str(batch_size),
            str(self.weight_decay),
            str(True if len(resume) > 0 else False)
        )
        self.model_dir_B = self.model_dir + "_B"
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if not os.path.exists("./prm_cache"):
                os.mkdir("./prm_cache")
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir))
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir_B)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir_B))
        self.model_dir = "./prm_cache/" + self.model_dir
        self.model_dir_B = "./prm_cache/" + self.model_dir_B
        ''' SET working dirs '''

        ''' SET working dir for the loss model '''
        self.loss_model_save_path = os.path.join(self.model_dir, "loss_model")
        self.loss_model_save_path_B = os.path.join(self.model_dir_B, "loss_model")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if os.path.exists(self.loss_model_save_path):
                print(f"=== REMOVE the existing loss model file from {self.loss_model_save_path} ===")
                import shutil
                shutil.rmtree(self.loss_model_save_path)

            if not os.path.exists(self.loss_model_save_path):
                os.mkdir(self.loss_model_save_path)

            if os.path.exists(self.loss_model_save_path_B):
                print(f"=== REMOVE the existing loss model file from {self.loss_model_save_path_B} ===")
                import shutil
                shutil.rmtree(self.loss_model_save_path_B)

            if not os.path.exists(self.loss_model_save_path_B):
                os.mkdir(self.loss_model_save_path_B)
        ''' SET working dir for the loss model '''

        ''' GET model '''
        self.args.lr_scaler = hvd.size()

        self.args.loss_model_save_path = self.loss_model_save_path
        self.model = InstSegNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args  # args
        )
        self.model.cuda()

        self.args.loss_model_save_path = self.loss_model_save_path_B
        self.model_B = InstSegNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args  # args
        )
        self.model_B.cuda()
        # self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
        ''' GET model '''

        ''' Set datasets '''
        # DATASET ROOT & NUMBER OF MASKS
        self.dataset_root = self.args.dataset_root
        self.nmasks = int(self.args.pred_nmasks)

        # DATASETS for train & val & test
        self.train_dataset = self.args.train_dataset
        self.val_dataset = self.args.val_dataset
        self.test_dataset = self.args.test_dataset

        self.split_type = self.args.split_type
        self.split_train_test = self.args.split_train_test

        ### GET test data-types for all categories in test setting ####
        self.pure_test = self.args.pure_test
        pure_test_types = self.args.pure_test_types.split(";")
        self.pure_test_types = [str(tpt) for tpt in pure_test_types]
        ### GET test data-types for all categories in test setting ####

        # TRAIN & VAL & TEST PartNet types # for inst-seg only
        partnet_train_types = self.args.partnet_train_types.split(";")
        self.partnet_train_types = [str(tpt) for tpt in partnet_train_types]
        partnet_val_types = self.args.partnet_val_types.split(";")
        self.partnet_val_types = [str(tpt) for tpt in partnet_val_types]
        partnet_test_types = self.args.partnet_test_types.split(";")
        self.partnet_test_types = [str(tpt) for tpt in partnet_test_types]
        # TRAIN & VAL & TEST PartNet types # for inst-seg only

        #
        train_prim_types = self.args.train_prim_types.split(",")
        self.train_prim_types = [int(tpt) for tpt in train_prim_types]
        test_prim_types = self.args.test_prim_types.split(",")
        self.test_prim_types = [int(tpt) for tpt in test_prim_types]

        #### SET pure test setting ####
        self.test_performance = self.args.test_performance
        #### SET pure test setting ####

        self.best_eval_acc = -999.0

        self.partnet_test_types = ['Scissors', 'Faucet', 'Door Set', 'Lamp']

        with FileLock(os.path.expanduser("~/.horovod_lock")):
            self.train_set = PartSegmentationMetaInfoDataset(
                root=self.dataset_root, split='train', npoints=512, nmask=args.nmasks, relrot=True,
                load_file_name="tot_part_motion_meta_info.npy",
                shape_types=["03642806", "03636649", "02691156", "03001627", "02773838", "02954340", "03467517",
                             "03790512",
                             "04099429", "04225987", "03624134", "02958343", "03797390", "03948459", "03261776",
                             "04379243"],
                # shape_types=["04379243", "03001627"],
                split_data=None,
                part_net_seg=False, args=args
            )

            self.eval_set = PartSegmentationMetaInfoDataset(
                root=self.dataset_root, split='val', npoints=512, nmask=args.nmasks, relrot=True,
                load_file_name="tot_part_motion_meta_info.npy",
                shape_types=["03642806", "03636649", "02691156", "03001627", "02773838", "02954340", "03467517",
                             "03790512",
                             "04099429", "04225987", "03624134", "02958343", "03797390", "03948459", "03261776",
                             "04379243"],
                part_net_seg=False, args=args
            )

            if not (self.args.test_performance and not self.args.beam_search):
                self.partnet_train_set = PartSegmentationMetaInfoDataset(
                    root=self.dataset_root, split='train', npoints=512, nmask=10, relrot=True,
                    load_file_name="tot_part_motion_meta_info.npy",
                    shape_types=self.partnet_test_types,
                    split_data=None, part_net_seg=True, partnet_split=False, args=args # True
                )

                self.partnet_test_set = PartSegmentationMetaInfoDataset(
                    root=self.dataset_root, split='tst', npoints=512, nmask=10, relrot=True,
                    load_file_name="tot_part_motion_meta_info.npy",
                    shape_types=self.partnet_test_types,
                    split_data=None, part_net_seg=True, partnet_split=False, args=args # True
                )
        print("Loaded")
        ''' Set datasets '''

        ''' Set dataset samplers '''
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set, num_replicas=hvd.size(), rank=hvd.rank())

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            self.eval_set, num_replicas=hvd.size(), rank=hvd.rank())

        if not (self.args.test_performance and not self.args.beam_search):
            self.partnet_train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.partnet_train_set, num_replicas=hvd.size(), rank=hvd.rank())

            self.partnet_val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.partnet_test_set, num_replicas=hvd.size(), rank=hvd.rank())
        ''' Set dataset samplers '''

        ''' Set dataloaders '''
        self.train_loader = data.DataLoader(
            self.train_set, batch_size=self.batch_size,
            sampler=self.train_sampler, **kwargs)

        self.val_loader = data.DataLoader(
            self.eval_set, batch_size=self.batch_size,
            sampler=self.val_sampler, **kwargs)

        if not (self.args.test_performance and not self.args.beam_search):
            self.partnet_train_loader = data.DataLoader(
                self.partnet_train_set, batch_size=self.batch_size,
                sampler=self.partnet_train_sampler, **kwargs)

            self.partnet_val_loader = data.DataLoader(
                self.partnet_test_set, batch_size=self.batch_size,
                sampler=self.partnet_val_sampler, **kwargs)
        ''' Set dataloaders '''

        ''' SET optimizers '''
        lr_scaler = hvd.size()
        self.lr_scaler = lr_scaler

        if hvd.nccl_built():
            lr_scaler = hvd.local_size()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.init_lr * lr_scaler,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay
        )

        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.model.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.optimizer_B = torch.optim.Adam(
            self.model_B.parameters(),
            lr=self.init_lr * lr_scaler,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        self.optimizer_B = hvd.DistributedOptimizer(
            self.optimizer_B,
            named_parameters=self.model_B.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        hvd.broadcast_parameters(self.model_B.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer_B, root_rank=0)
        ''' SET optimizers '''

        ''' SET related sampling distributions '''
        # number of grouping operations, binary operations and unary operations
        self.nn_grp_opers = args.nn_grp_opers
        self.nn_binary_opers = args.nn_binary_opers
        self.nn_unary_opers = args.nn_unary_opers
        self.nn_in_feats = args.nn_in_feats

        ''' SET sampling tree '''
        if self.args.v2_tree:
            sampling_tree = DistributionTreeNodeV2
        else:
            sampling_tree = DistributionTreeNode

        if not args.debug and not args.pure_test:
            if not args.debug_arch:
                print(f"Use less layers = {self.args.less_layers}")
                self.sampling_tree_rt = sampling_tree(cur_depth=0 if (self.args.less_layers is False) else 1, nn_grp_opers=self.nn_grp_opers, nn_binary_opers=self.nn_binary_opers, nn_unary_opers=self.nn_unary_opers, nn_in_feat=self.nn_in_feats, args=args)

            self.arch_dist = DistributionTreeNodeArch(cur_depth=0, tot_layers=3, nn_conv_modules=4, device=None,
                                                      args=None)
        ''' SET related sampling distributions '''

        self.in_model_loss_model = args.in_model_loss_model
        if not self.args.in_model_loss_model:
            ''' SET loss model '''
            self.loss_model = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                      lr_scaler=lr_scaler, init_lr=self.init_lr,
                                                      weight_decay=self.weight_decay,
                                                      loss_model_save_path=self.loss_model_save_path,
                                                      in_rep_dim=self.map_feat_dim if not self.args.use_spfn else 128,
                                                      nn_inter_loss=self.nn_inter_loss,
                                                      args=args
                                                      )
            # SET optimizer for loss model
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

            hvd.broadcast_optimizer_state(self.head_optimizer, root_rank=0)
            # todo: can we broadcast a model containing a list of modules?
            hvd.broadcast_parameters(self.loss_model.state_dict(), root_rank=0)
            ''' SET loss model '''

    def adjust_learning_rate_by_factor(self, scale_factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale_factor

    def from_dist_dict_to_dist_tsr(self, dist_dict):
        # uop gop bop oper chd_left chd_right

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
            if oper_tsr.size(0) < maxx_sz:
                pad_oper_tsr = torch.cat([oper_tsr, torch.full((maxx_sz - oper_tsr.size(0), ), fill_value=-2, dtype=torch.long).cuda()], dim=0)
            else:
                pad_oper_tsr = oper_tsr
            pad_oper_tsr_list.append(pad_oper_tsr.unsqueeze(0))

        if len(pad_oper_tsr_list) > 0:
            pad_oper_tsr = torch.cat(pad_oper_tsr_list, dim=0)
        else:
            pad_oper_tsr = torch.full((1,200), fill_value=-2, dtype=torch.long).cuda()

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

        if len(clean_dist_list) > 0:
            dist_dict = self.from_oper_list_to_oper_dict(clean_dist_list)
        else:
            dist_dict = None

        return dist_dict

    def from_dist_tsr_to_dist_dict_list(self, dist_tsr):
        n_samples = dist_tsr.size(0)
        dist_dict_list = []
        for i in range(n_samples):
            cur_dict = self.from_dist_tsr_to_dist_dict(dist_tsr[i])
            if cur_dict is not None:
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
                if sampled_loss_dict is not None:
                    res.append(sampled_loss_dict)
        return res

    def update_dist_params_by_sampled_oper_dists_list_accum(self, oper_dist_list, rewards, baseline, lr):
        n_samples = len(oper_dist_list)
        ''' ACCUMULATE operator selection & transformation selection distributions '''
        self.sampling_tree_rt.update_sampling_dists_by_sampled_dicts(oper_dist_list, rewards, baseline, lr=lr)

    def print_dist_params(self):
        print("=== Distribution parameters ===")
        self.sampling_tree_rt.print_params()

    def print_dist_params_to_file_arch(self):
        self.arch_dist.collect_params_and_save(os.path.join(self.model_dir, "dist_params_arch.npy"))

    def print_dist_params_to_file(self, param_file_name="dist_params.npy"):
        if self.args.v2_tree:
            self.sampling_tree_rt.collect_params_and_save(os.path.join(self.model_dir, param_file_name))
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
        if momask.size(1) > momask.size(2):
            momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        # gt_conf = torch.where(gt_conf > 0, 1, 0).float()
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf

    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def get_nn_segmentations(self, batch_inst_seg):
        bz = batch_inst_seg.size(0)
        tot_nn_segs = 0
        for i in range(bz):
            cur_inst_seg = batch_inst_seg[i]
            cur_seg_nn = int(torch.max(cur_inst_seg).item()) + 1
            tot_nn_segs += cur_seg_nn
        return tot_nn_segs

    def _train_one_epoch(
            self, epoch,
            conv_select_types=[0, 0, 0, 0, 0],
            loss_selection_dict={},
            desc="train",
            cur_model=None,
            cur_loaders=None,
            cur_samplers=None,
            cur_optimizer=None
        ):
        # conv_select_types = self.conv_select_types
        cur_model = self.model if cur_model is None else cur_model
        cur_model.train()
        if not self.in_model_loss_model:
            self.loss_model.train()

        loss_list = []
        loss_nn = []

        cur_loaders = [self.train_loader] if cur_loaders is None else cur_loaders
        train_bars = [tqdm(cur_loader) for cur_loader in cur_loaders]
        cur_samplers = [self.train_sampler] if cur_samplers is None else cur_samplers
        iouvalue = []

        avg_recall = []

        gt_loss = []

        tot_seg_nns = []

        totn = 0

        for train_bar in train_bars:
            for i_batch, batch_data in enumerate(train_bar):
                try:
                    batch_pos = batch_data['pc1_af_glb']
                except:
                    print(f"Type: {type(batch_data)}")
                    print(batch_data)
                    raise ValueError("zzz")
                    continue


                batch_flow = batch_data['flow12']
                batch_momasks = batch_data['motion_seg_masks']



                #
                batch_inst_seg = torch.argmax(batch_momasks, dim=-1) # assume momask.size = bz x N x nmasks
                # print(f"flow: {batch_flow.size()}, masks: {batch_momasks.size()}, batch_inst_seg: {batch_inst_seg.size()}")

                cur_batch_nn_seg = self.get_nn_segmentations(batch_inst_seg)

                tot_seg_nns.append(cur_batch_nn_seg)

                if batch_pos.size(0) == 1:
                    continue

                batch_pos = batch_pos.float().cuda()
                if batch_pos.size(-1) > 3:
                    batch_Rs = batch_pos[:, :, 3:12]
                    batch_ts = batch_pos[:, :, 12:]
                    batch_pos = batch_pos[:, :, :3]

                batch_inst_seg = batch_inst_seg.long().cuda()
                # batch_momasks = get_one_hot(batch_inst_seg, 200)
                batch_momasks = batch_momasks.float().cuda()
                batch_flow = batch_flow.float().cuda()

                # batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
                bz, N = batch_pos.size(0), batch_pos.size(1)

                feats = {'flow': batch_flow}
                rt_values = cur_model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model if not self.in_model_loss_model else cur_model.intermediate_loss if self.args.add_intermediat_loss else None
                )

                seg_pred, gt_l, pred_conf, losses, fps_idx = rt_values

                if "iou" in losses:
                    iou_value = losses["iou"]
                    cur_avg_recall = 0.0
                    seg_loss = None
                else:
                    # bz x N x nmasks
                    momasks_sub = batch_momasks

                    batch_momasks, batch_conf = self.get_gt_conf(momasks_sub)

                    batch_momasks = batch_momasks

                    # print(f"seg_pred: {seg_pred.size()}, batch_momasks: {batch_momasks.size()}, batch_conf: {batch_conf.size()}")

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

                loss = 0.0

                neg_iou_loss = -1.0 * iou_value

                loss += neg_iou_loss

                if self.args.add_intermediat_loss:
                    loss += gt_l.mean()

                if seg_loss is not None and self.args.with_conf_loss:
                    loss += seg_loss.mean()

                if self.args.with_rt_loss and 'pred_Rs' in losses and 'pred_ts' in losses:
                    pred_Rs_loss = torch.mean(torch.sum((losses['pred_Rs'] - batch_Rs) ** 2, dim=-1))
                    pred_ts_loss = torch.mean(torch.sum((losses['pred_ts'] - batch_ts) ** 2, dim=-1))
                    loss += self.args.r_loss_ratio * pred_Rs_loss + self.args.t_loss_ratio * pred_ts_loss
                try:
                    gt_loss.append(gt_l.detach().cpu().item() * bz)
                except:
                    gt_loss.append(gt_l * bz)

                if not self.in_model_loss_model:
                    self.head_optimizer.zero_grad()

                cur_optimizer.zero_grad()
                # self.optimizer.zero_grad()

                loss.backward()

                cur_optimizer.step()
                # self.loss_model.head_optimizer.step()
                if not self.in_model_loss_model:
                    self.head_optimizer.step()

                loss_list += [loss.detach().cpu().item() * bz]
                loss_nn.append(bz)
                try:
                    iouvalue.append(-neg_iou_loss.detach().cpu().item() * bz)
                except:
                    iouvalue.append(-neg_iou_loss * bz)

                train_bar.set_description(
                    'Train Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Iou:{:.3f} AvgRecall:{:.3f} AvgSegNN:{:.3f}'.format(
                        epoch + 1, 200, float(sum(loss_list) / sum(loss_nn)),
                        float(sum(gt_loss)) / float(sum(loss_nn)),
                        float(sum(iouvalue) / sum(loss_nn)),
                        float(sum(avg_recall) / sum(loss_nn)),
                        float(float(sum(tot_seg_nns)) / float(sum(loss_nn))),
                ))

        avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
        avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
        avg_iou = float(sum(iouvalue)) / float(sum(loss_nn))
        avg_recall = float(sum(avg_recall)) / float(sum(loss_nn))
        avg_seg_nns = float(sum(tot_seg_nns)) / float(sum(loss_nn))
        avg_loss = self.metric_average(avg_loss, 'tr_loss')
        avg_gt_loss = self.metric_average(avg_gt_loss, 'tr_gt_loss')
        avg_iou = self.metric_average(avg_iou, 'tr_iou')
        avg_recall = self.metric_average(avg_recall, 'tr_avg_recall')
        avg_seg_nns = self.metric_average(avg_seg_nns, 'tr_avg_seg_nns')

        if hvd.rank() == 0:
            # LOG to file
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("Train Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall:{:.3f} AvgSegNN:{:.3f}".format(
                    epoch + 1, avg_loss, avg_gt_loss,
                    avg_iou,
                    avg_recall,
                    avg_seg_nns
                ) + "\n")
                wf.close()

            # LOGGING
            logging.info("Train Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall:{:.3f} AvgSegNN:{:.3f}".format(
                epoch + 1, avg_loss, avg_gt_loss,
                avg_iou,
                avg_recall,
                avg_seg_nns
            ))
        return avg_iou, avg_gt_loss #  avg_recall

    def _test(
            self, epoch, desc="val",
            conv_select_types=[0, 0, 0],
            loss_selection_dict=[],
            cur_loader=None,
            cur_model=None,
        ):
        cur_model = self.model if cur_model is None else self.model
        cur_model.eval()

        if not self.in_model_loss_model:
            self.loss_model.eval()

        with torch.no_grad():

            totn = 0

            if cur_loader is None:
                if desc == "val":
                    cur_loader = self.val_loader # self.partnet_val_loader
                elif desc == "test":
                    cur_loader = self.test_loader
                elif desc == "partnet_val":
                    cur_loader = self.partnet_val_loader
                else:
                    raise ValueError(f"Unrecognized desc: {desc}")

            # cur_loader = self.test_loader

            loss_list = []
            loss_nn = []
            test_bar = tqdm(cur_loader)
            iouvalue = []

            gt_loss = []

            avg_recall = []
            tot_seg_nns = []

            for batch_data in test_bar:

                batch_pos = batch_data['pc1_af_glb']
                batch_flow = batch_data['flow12']
                batch_momasks = batch_data['motion_seg_masks']

                batch_inst_seg = torch.argmax(batch_momasks, dim=-1)  # assume momask.size = bz x N x nmasks

                cur_batch_nn_seg = self.get_nn_segmentations(batch_inst_seg)

                tot_seg_nns.append(cur_batch_nn_seg)

                if batch_pos.size(0) == 1:
                    continue
                if batch_pos.size(-1) > 3:
                    batch_pos = batch_pos[..., :3]

                batch_pos = batch_pos.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                # batch_momasks = get_one_hot(batch_inst_seg, 200)
                batch_momasks = batch_momasks.float().cuda()
                batch_flow = batch_flow.float().cuda()

                # batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
                bz, N = batch_pos.size(0), batch_pos.size(1)

                feats = {'flow': batch_flow}

                rt_values = cur_model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model if not self.in_model_loss_model else cur_model.intermediate_loss if self.args.add_intermediat_loss else None
                )

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
                    # momasks_sub = get_one_hot(batch_inst_seg, 30)
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
                        # print(torch.mean(batch_conf.sum(-1)))
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

                loss = 0.0
                loss += neg_iou_loss

                if self.args.add_intermediat_loss:
                    loss += gt_l.mean()

                if seg_loss is not None and self.args.with_conf_loss:
                    loss += seg_loss.mean()

                loss_list += [loss.detach().cpu().item() * bz]

                try:
                    gt_loss.append(gt_l.detach().cpu().item() * bz)
                except:
                    gt_loss.append(gt_l * bz)

                loss_nn.append(bz)
                try:
                    iouvalue.append(-neg_iou_loss.detach().cpu().item() * bz)
                except:
                    iouvalue.append(-neg_iou_loss * bz)

                test_bar.set_description(
                    # 'Test_{} Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Losses:{} Iou:{:.3f} AvgRecall:{:.3f} Recode Iou: {} '.format(
                    'Test_{} Epoch: [{}/{}] Loss:{:.3f} GT_L:{:.3f} Iou:{:.3f} AvgRecall:{:.3f} AvgSegNN:{:.3f}'.format(
                        desc,
                        epoch + 1, 200, float(sum(loss_list) / sum(loss_nn)),
                        float(sum(gt_loss)) / float(sum(loss_nn)),
                        # str(losses),
                        float(sum(iouvalue) / sum(loss_nn)),
                        float(sum(avg_recall) / sum(loss_nn)),
                        float(float(sum(tot_seg_nns)) / float(sum(loss_nn))),
                        # str(thr_to_recall)
                ))

            # return 0, 0

            avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
            avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
            avg_iou = float(sum(iouvalue)) / float(sum(loss_nn))
            avg_recall = float(sum(avg_recall)) / float(sum(loss_nn))
            avg_seg_nns = float(sum(tot_seg_nns)) / float(sum(loss_nn))
            avg_loss = self.metric_average(avg_loss, 'loss')
            avg_gt_loss = self.metric_average(avg_gt_loss, 'gt_loss')
            avg_iou = self.metric_average(avg_iou, 'iou')
            avg_recall = self.metric_average(avg_recall, 'avg_recall')
            avg_seg_nns = self.metric_average(avg_seg_nns, 'tr_avg_seg_nns')

            if hvd.rank() == 0:
                with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                    wf.write("Test_{} Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall: {:.3f} AvgSegNN:{:.3f}".format(
                        desc,
                        epoch + 1,
                        avg_loss,
                        avg_gt_loss,
                        avg_iou,
                        avg_recall,
                        avg_seg_nns
                    ) + "\n")
                    wf.close()

                logging.info("Test_{} Epoch: {:d}, loss: {:.4f} GT_L: {:.4f} Iou: {:.3f} AvgRecall: {:.3f} AvgSegNN:{:.3f}".format(
                    desc,
                    epoch + 1,
                    avg_loss,
                    avg_gt_loss,
                    avg_iou,
                    avg_recall,
                    avg_seg_nns
                ))
            return avg_iou, avg_gt_loss

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "checkpoint_current.pth".format(epoch)))

    ''' SEARCH supervisions '''
    def rein_train_search_loss(
        self, base_epoch,
        base_arch_select_type,  # a list of selected baseline architecture
        baseline,
    ):
        eps_training_epochs = 1
        n_models_per_eps = 4
        # tot_eps = 7 # 5
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
                    k=self.nn_inter_loss, baseline=False)
                if hvd.rank() == 0:
                    logging.info(f"cur_selected_loss_dict = {cur_selected_loss_dict_list}")
                # cur_selected_loss_tsr = self.from_dist_dict_to_dist_tsr(cur_selected_loss_dict)
                cur_selected_loss_tsr = self.form_dist_dict_list_to_dist_tsr(cur_selected_loss_dict_list)

                cur_selected_loss_tsr = hvd.broadcast(cur_selected_loss_tsr, root_rank=0)

                # cur_selected_loss_dict = self.from_dist_tsr_to_dist_dict(cur_selected_loss_tsr)
                cur_selected_loss_dict_list = self.from_dist_tsr_to_dist_dict_list(cur_selected_loss_tsr)

                if hvd.rank() == 0:
                    logging.info(f"Sampled loss dict: {cur_selected_loss_dict_list}")
                if i_eps >= 0:
                    # sampled_ratios.append(cur_model_sampled_selected_types.unsqueeze(0))
                    # sampled_rs.append(cur_model_sampled_r)
                    sampled_loss_dicts.append(cur_selected_loss_dict_list)

                    ''' LOAD model '''
                    self.model.load_state_dict(
                        torch.load(os.path.join(self.model_dir, "REIN_init_saved_model.pth"),
                                   map_location='cpu'
                                   )
                    )
                    self.model.cuda()  # .to(self.device)

                    self.model_B.load_state_dict(
                        torch.load(os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"),
                                   map_location='cpu'
                                   )
                    )
                    self.model_B.cuda()  # .to(self.device)
                    ''' LOAD model '''

                    self.model.intermediate_loss.save_init_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list)

                    self.model.intermediate_loss.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                                        init_lr=self.init_lr,
                                                                                        weight_decay=self.weight_decay)

                    self.model_B.intermediate_loss.save_init_head_optimizer_by_operation_dicts(
                        cur_selected_loss_dict_list)

                    self.model_B.intermediate_loss.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                                        init_lr=self.init_lr,
                                                                                        weight_decay=self.weight_decay)

                    best_test_acc = 0.0

                    for i_iter in range(eps_training_epochs):
                        train_acc, train_gt_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model,
                            cur_loaders=[self.train_loader],
                            cur_samplers=[self.train_sampler],
                            cur_optimizer=self.optimizer
                        )

                        val_acc, val_gt_loss = self._test(
                            base_epoch + i_iter + 1, desc="partnet_val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model,
                            cur_loader=self.partnet_val_loader,
                            # r=cur_model_sampled_r
                        )
                        # val_acc = test_acc
                        best_test_acc += val_acc - train_acc

                    for i_iter in range(eps_training_epochs):
                        train_acc, train_gt_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_B,
                            cur_loaders=[self.partnet_train_loader],
                            cur_samplers=[self.partnet_train_sampler],
                            cur_optimizer=self.optimizer_B
                        )

                        val_acc, val_gt_loss = self._test(
                            base_epoch + i_iter + 1, desc="partnet_val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_B,
                            cur_loader=self.val_loader,
                            # r=cur_model_sampled_r
                        )
                        best_test_acc += val_acc - train_acc

                    best_test_acc /= 2
                    rewards.append(best_test_acc)

                    if hvd.rank() == 0:
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_dir, f"REIN_saved_model_{i_model}.pth"))

                        self.model.intermediate_loss.save_head_optimizer_by_operation_dicts(
                            cur_selected_loss_dict_list, i_model)

                        torch.save(self.model_B.state_dict(),
                                   os.path.join(self.model_dir_B, f"REIN_saved_model_{i_model}.pth"))

                        self.model_B.intermediate_loss.save_head_optimizer_by_operation_dicts(
                            cur_selected_loss_dict_list, i_model)

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
                # self.update_dist_params_by_sampled_oper_dists(sampled_loss_dicts, rewards=rewards, baseline=baseline, lr=self.init_lr)
                # sampled loss dictionaries
                # #### update distribution parameters by sampled operator dictionaries and rewards
                self.update_dist_params_by_sampled_oper_dists_list_accum(sampled_loss_dicts, rewards=rewards,
                                                                         baseline=baseline, lr=self.init_lr)
                ''' PRINT loss sampling distribution related parameters '''
                print("Distribution parameters after update: ")
                # self.print_dist_params()
                self.print_dist_params_to_file()
                if (base_epoch + i_eps) % 20 == 0:
                    # print dist params for space evolution evluation
                    self.print_dist_params_to_file(param_file_name=f"dist_params_{base_epoch + i_eps}.npy")
                ''' PRINT loss sampling distribution related parameters '''

            ''' LOAD best model '''
            logging.info(f"Loading from best model idx = {best_model_idx}")
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, f"REIN_saved_model_{best_model_idx}.pth"),
                           map_location="cpu"
                           ))
            self.model.cuda()

            self.model_B.load_state_dict(
                torch.load(os.path.join(self.model_dir_B, f"REIN_saved_model_{best_model_idx}.pth"),
                           map_location="cpu"
                           ))
            self.model_B.cuda()
            ''' LOAD best model '''

            ''' SAVE as init_model '''
            if hvd.rank() == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))
                self.model.intermediate_loss.save_head_optimizer_by_operation_dicts(
                    sampled_loss_dicts[best_model_idx])

                torch.save(self.model_B.state_dict(), os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"))
                self.model_B.intermediate_loss.save_head_optimizer_by_operation_dicts(
                    sampled_loss_dicts[best_model_idx])
            best_test_acc = 0.0
            base_epoch += eps_training_epochs

            ''' GET baseline loss dict '''
            baseline_loss_dict = self.sample_intermediate_representation_generation_k_list(
                k=self.nn_inter_loss, baseline=True)

            ''' BROADCAST baseline loss dict '''
            baseline_loss_tsr = self.form_dist_dict_list_to_dist_tsr(baseline_loss_dict)
            baseline_loss_tsr = hvd.broadcast(baseline_loss_tsr, root_rank=0)
            baseline_loss_dict = self.from_dist_tsr_to_dist_dict_list(baseline_loss_tsr)

            ### LOAD loss model state dictionary for current main model ###
            self.model.intermediate_loss.save_init_head_optimizer_by_operation_dicts(baseline_loss_dict,)
            # self.model.cuda()
            self.model.intermediate_loss.load_head_optimizer_by_operation_dicts(baseline_loss_dict,
                                                                                init_lr=self.init_lr,
                                                                                weight_decay=self.weight_decay)
            self.model.cuda()

            ### LOAD loss model state dictionary for current main model ###
            self.model_B.intermediate_loss.save_init_head_optimizer_by_operation_dicts(baseline_loss_dict, )
            # self.model.cuda()
            self.model_B.intermediate_loss.load_head_optimizer_by_operation_dicts(baseline_loss_dict,
                                                                                init_lr=self.init_lr,
                                                                                weight_decay=self.weight_decay)
            self.model_B.cuda()

            for i_iter in range(eps_training_epochs):
                train_acc, train_gt_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loaders=[self.train_loader],
                    cur_samplers=[self.train_sampler],
                    cur_optimizer=self.optimizer
                    # r=baseline_r
                )

                val_acc, val_gt_loss = self._test(
                    base_epoch + i_iter + 1, desc="partnet_val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loader=self.partnet_val_loader
                )
                best_test_acc += val_acc - train_acc

            for i_iter in range(eps_training_epochs):
                train_acc, train_gt_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_B,
                    cur_loaders=[self.partnet_train_loader],
                    cur_samplers=[self.partnet_train_sampler],
                    cur_optimizer=self.optimizer_B
                    # r=baseline_r
                )

                val_acc, val_gt_loss = self._test(
                    base_epoch + i_iter + 1, desc="partnet_val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_B,
                    cur_loader=self.val_loader
                )
                best_test_acc += val_acc - train_acc

            baseline = best_test_acc / 2
            # logging.info(f"rank = {hvd.rank()}, Current baseline = {baseline}\n")
            if hvd.rank() == 0:
                logging.info(f"Current baseline = {baseline}")
                with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                    wf.write(f"Current baseline = {baseline}" + "\n")
                    wf.write(f"Baseline model dictionary selection = {baseline_loss_dict}" + "\n")
                    wf.close()
            final_baseline = baseline
            final_baseline_loss_dict = baseline_loss_dict

        return final_baseline, final_baseline_loss_dict

    def greedy_search(self, base_dict=[], max_nn=3, keep_nn=2, conv_select_types=[0, 0, 0]):
        unary_aa = []
        unary_bb = []

        binary_dicts = []
        for i in range(len(unary_aa)):
            for j in range(i, len(unary_bb)):
                cur_dicts = [unary_aa[i], unary_bb[j]]
                binary_dicts.append(cur_dicts)

        val_training_epochs = 25
        maxx_val_acc = -999.0
        val_accs = []

        # for j, cur_dict in enumerate(all_dicts):
        for j, cur_dict in enumerate(binary_dicts):
            # current binary dict
            cur_feed_dicts = cur_dict
            ''' LOAD related head weights of the correct shape '''
            if self.args.add_intermediat_loss:
                self.model.intermediate_loss.load_head_optimizer_by_operation_dicts(
                    cur_feed_dicts, init_lr=self.init_lr, weight_decay=self.weight_decay)
                self.model_B.intermediate_loss.load_head_optimizer_by_operation_dicts(
                    cur_feed_dicts, init_lr=self.init_lr, weight_decay=self.weight_decay)
            #### LOAD model's weights ####
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, "REIN_init_saved_model.pth"), map_location="cpu")
            )
            self.model_B.load_state_dict(
                torch.load(os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"), map_location="cpu")
            )
            cur_maxx_val_acc = -999.0

            for i in range(val_training_epochs):
                train_acc, train_recall = self._train_one_epoch(
                    i + 1,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts, desc="train",
                    cur_model=self.model,
                    cur_loaders=[self.train_loader],
                    cur_optimizer=self.optimizer,
                    cur_samplers=[self.train_sampler]
                )
                # return

                val_acc, _ = self._test(
                    i + 1, desc="partnet_val",
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model,
                    cur_loader=self.partnet_val_loader
                )

                train_acc2, train_recall2 = self._train_one_epoch(
                    i + 1,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts, desc="train",
                    cur_model=self.model_B,
                    cur_loaders=[self.partnet_train_loader],
                    cur_optimizer=self.optimizer_B,
                    cur_samplers=[self.partnet_train_sampler]
                )

                val_acc2, _ = self._test(
                    i + 1, desc="partnet_val",
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model_B,
                    cur_loader=self.val_loader
                )

                cur_maxx_val_acc = max(cur_maxx_val_acc, val_acc + val_acc2)

            val_accs.append((j, cur_maxx_val_acc))
        #### sort validation accs ####
        val_accs = sorted(val_accs, key=lambda kk: kk[1], reverse=True)
        #### GET topk dicts ####
        topk_dicts = [binary_dicts[kk[0]] for kki, kk in enumerate(val_accs[:keep_nn])]
        tot_dicts_ordered = [binary_dicts[kk[0]] for kki, kk in enumerate(val_accs)]

        if hvd.rank() == 0:
            with open(os.path.join(self.model_dir, "selected_dicts.txt"), "a") as wf:
                # wf.write(f"{len(base_dict) + 1}:\n")
                wf.write(f"{1 + 1}:\n")
                for cur_rt_dict, cur_val_acc_item in zip(tot_dicts_ordered, val_accs):
                    cur_val_acc = cur_val_acc_item[1]
                    # wf.write(f"{str(base_dict + [cur_rt_dict])}\t{cur_val_acc}\n")
                    wf.write(f"{str(base_dict + cur_rt_dict)}\t{cur_val_acc}\n")
                wf.close()
        return topk_dicts

    def train_all(self):
        # baseline_value = torch.tensor([1, 1, 0], dtype=torch.long)
        baseline_value = torch.tensor([0, 0, 0], dtype=torch.long)

        if self.args.beam_search:
            baseline_value = torch.tensor([1, 1, 0], dtype=torch.long)
            rt_dicts = self.greedy_search(base_dict=[], max_nn=3, keep_nn=2, conv_select_types=baseline_value.tolist())
            print(rt_dicts)
            return

        eps_training_epochs = 1
        tot_eps = 100
        ''' SAVE current model weights as initial weights '''
        if hvd.rank() == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))
            torch.save(self.model_B.state_dict(), os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"))

        ''' GET baseline loss dict (no loss is selected in baseline loss dict) '''

        baseline_loss_dict = []

        best_val_acc = 0.0
        best_test_acc = 0.0

        baseline_value = torch.tensor([1, 1, 1], dtype=torch.long)

        not_improved_num_epochs = 0

        ''' LOAD model '''
        if self.args.resume != "":
            logging.info(f"Loading model from {self.args.resume}")
            # resume
            ori_dict = torch.load(os.path.join(self.args.resume, "REIN_best_saved_model.pth"), map_location='cpu')
            part_dict = dict()
            model_dict = self.model.state_dict()
            for k in ori_dict:
                # if "feat_conv_net_trans_pred" not in k:
                if k in model_dict:
                    v = ori_dict[k]
                    part_dict[k] = v
            model_dict.update(part_dict)
            self.model.load_state_dict(model_dict)

            self.model.cuda()

        if self.test_performance:
            baseline_loss_dict = []

            if len(baseline_loss_dict) > 0:
                if not self.in_model_loss_model:
                    ''' LOAD loss model (head & optimizer) '''
                    if self.args.resume != "":
                        logging.info(f"Set loss model save path of self.loss_model save path to {self.args.resume}")
                        self.loss_model.loss_model_save_path = os.path.join(self.args.resume, "loss_model")

                    self.loss_model.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                           weight_decay=self.weight_decay)
                else:
                    if self.args.resume != "" and self.args.add_intermediat_loss:
                        self.model.intermediate_loss.loss_model_save_path = os.path.join(self.args.resume, "loss_model")
                    ''' LOAD related head weights of the correct shape '''
                    if self.args.add_intermediat_loss:
                        self.model.intermediate_loss.save_init_head_optimizer_by_operation_dicts(baseline_loss_dict, )
                        self.model.intermediate_loss.load_head_optimizer_by_operation_dicts(
                            baseline_loss_dict, init_lr=self.init_lr, weight_decay=self.weight_decay)

            eps_training_epochs = 400

        best_test_val_acc = 0.0
        best_val_acc = 0.0
        for i_iter in range(eps_training_epochs):
            train_acc, train_gt_loss = self._train_one_epoch(
                i_iter + 1,
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                desc="train",
                cur_model=self.model,
                cur_loaders=[self.train_loader],
                cur_samplers=[self.train_sampler],
                cur_optimizer=self.optimizer
            )

            val_acc, val_gt_loss = self._test(
                i_iter + 1, desc="val", # "partnet_val",
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model,
                cur_loader=self.partnet_val_loader,
                # r=cur_model_sampled_r
            )

            best_val_acc += val_acc - train_acc

        for i_iter in range(eps_training_epochs):
            train_acc, train_gt_loss = self._train_one_epoch(
                i_iter + 1,
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict, desc="train",
                cur_model=self.model_B,
                cur_loaders=[self.partnet_train_loader],
                cur_samplers=[self.partnet_train_sampler],
                cur_optimizer=self.optimizer_B
            )

            # if not self.test_performance:
            val_acc, val_gt_loss = self._test(
                i_iter + 1, desc="val",
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model_B,
                cur_loader=self.val_loader,
                # r=cur_model_sampled_r
            )

            best_val_acc += val_acc - train_acc

        baseline = best_val_acc / 2

        base_epoch = eps_training_epochs - 1
        if hvd.rank() == 0:
            logging.info(f"Baseline = {baseline}")

        each_search_epochs = self.sea_interval
        base_model_select_arch_types = baseline_value.tolist()

        for i_eps in range(tot_eps):
            ''' Supervision search '''
            baseline, baseline_loss_dict = self.rein_train_search_loss(
                base_epoch=base_epoch, base_arch_select_type=base_model_select_arch_types, baseline=baseline
            )
            base_epoch += each_search_epochs
            ''' Supervision search '''
