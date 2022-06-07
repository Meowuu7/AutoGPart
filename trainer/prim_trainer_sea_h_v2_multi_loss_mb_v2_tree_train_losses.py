import torch
import torch.nn as nn
# from model.pointnetpp_segmodel_sea import PointNetPPInstSeg
# from model.primitive_fitting_net_sea import PrimitiveFittingNet
from model.primitive_fitting_net_sea_v2 import PrimitiveFittingNet

# from datasets.Indoor3DSeg_dataset import Indoor3DSemSeg
# from datasets.instseg_dataset import InstSegmentationDataset
# from datasets.ABC_dataset import ABCDataset
from datasets.ANSI_dataset import ANSIDataset
from torch.nn import functional as F

from tqdm import tqdm
import os
from torch.utils import data
from model.utils import iou
import horovod.torch as hvd
import torch.multiprocessing as mp
from filelock import FileLock
# import time
import numpy as np
from model.utils import batched_index_select, calculate_acc
from .trainer_utils import MultiMultinomialDistribution
import logging
from model.loss_model_v5 import ComputingGraphLossModel
from .trainer_utils import get_masks_for_seg_labels, compute_param_loss, DistributionTreeNode, DistributionTreeNodeV2
from model.constants import *
from model.loss_utils import compute_embedding_loss, compute_normal_loss, compute_nnl_loss
# from model.utils import mean_shift
# from model.loss_utils import compute_miou, npy
# from model.abc_utils import compute_entropy, construction_affinity_matrix_normal
import psutil

class TrainerPrimitiveFitting(nn.Module):
    def __init__(self, dataset_root, num_points=512, batch_size=32, num_epochs=200, cuda=None, dataparallel=False,
                 use_sgd=False, weight_decay_sgd=5e-4,
                 resume="", dp_ratio=0.5, args=None):
        super(TrainerPrimitiveFitting, self).__init__()
        # n_layers: int, feat_dims: list, n_samples: list, n_class: int, in_feat_dim: int
        ''' Some basic configurations '''
        self.num_epochs = num_epochs
        if cuda is not None:
            self.device = torch.device("cuda:" + str(cuda)) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        # print("Before init")
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        # torch.cuda.manual_seed(42)

        # print("setting number threads...")
        torch.set_num_threads(5)

        kwargs = {'num_workers': 5, 'pin_memory': True}
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        ''' Some basic configurations '''

        ''' Some arguments '''
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
        self.stage = int(self.args.stage)
        self.nn_inter_loss = int(self.args.nn_inter_loss)
        self.nn_base_inter_loss = 0
        ''' Some arguments '''

        self.baseline_loss_dict = []
        tot_inter_feat_dim = 0
        for loss_dict in self.baseline_loss_dict:
            cur_feat_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(loss_dict, in_feat_dim=3, nn_binary_opers=args.nn_binary_opers)
            rt_multnn = 1
            for jj in cur_feat_dim:
                rt_multnn *= jj
            tot_inter_feat_dim += rt_multnn
        args.intermediat_feat_pred_dim = tot_inter_feat_dim
        self.base_loss_dicts = []

        n_samples = self.args.n_samples.split(",")
        self.n_samples = [int(ns) for ns in n_samples]
        self.map_feat_dim = int(self.args.map_feat_dim)
        self.n_layers = int(self.args.n_layers)
        assert self.n_layers == len(self.n_samples), f"Expect the times of down-sampling equal to n_layers, got n_layers = {self.n_layers}, times of down-sampling = {len(self.n_samples)}."
        assert self.n_layers == len(self.feat_dims), f"Expect the number of feature dims equal to n_layers, got n_layers = {self.n_layers}, number of dims = {len(self.feat_dims)}."

        ''' GET model & loss selection parameters '''
        conv_select_types = self.args.conv_select_types.split(",")
        self.conv_select_types = [int(tt) for tt in conv_select_types]
        self.point_feat_selection = int(self.args.point_feat_selection)
        self.point_geo_feat_selection = int(self.args.point_geo_feat_selection)
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

        self.model_B = PrimitiveFittingNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args
        )

        self.model_C = PrimitiveFittingNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args
        )

        ''' LOAD initial model weights from saved model weights '''
        if len(self.resume) != 0:
            print(f"Loading weights from {self.resume}")
            ori_dict = torch.load(self.resume, map_location='cpu')
            part_dict = dict()
            model_dict = self.model.state_dict()
            model_dict_B = self.model_B.state_dict()
            model_dict_C = self.model_C.state_dict()
            for k in ori_dict:
                # if "feat_conv_net_trans_pred" not in k:
                if k in model_dict:
                    v = ori_dict[k]
                    part_dict[k] = v
            model_dict.update(part_dict)
            self.model.load_state_dict(model_dict)
            model_dict_B.update(part_dict)
            self.model_B.load_state_dict(model_dict_B)
            model_dict_C.update(part_dict)
            self.model_C.load_state_dict(model_dict_C)

        self.model.cuda()
        self.model_B.cuda()
        self.model_C.cuda()
        #### GET model ####

        ''' Set datasets '''
        self.dataset_root = self.args.dataset_root
        self.nmasks = int(self.args.nmasks)
        self.train_dataset = self.args.train_dataset
        self.val_dataset = self.args.val_dataset
        self.test_dataset = self.args.test_dataset
        self.split_type = self.args.split_type
        self.split_train_test = self.args.split_train_test
        train_prim_types = self.args.train_prim_types.split(",")
        self.train_prim_types = [int(tpt) for tpt in train_prim_types]
        test_prim_types = self.args.test_prim_types.split(",")
        self.test_prim_types = [int(tpt) for tpt in test_prim_types]

        ''' Arg '''
        self.test_performance = self.args.test_performance

        with FileLock(os.path.expanduser("~/.horovod_lock")):
            try:
                ansi_train_prim_types_1 = self.args.ansi_c_tr_prim_1.split(",")
                self.ansi_train_prim_types_1 = [int(tpt) for tpt in ansi_train_prim_types_1]
            except:
                ansi_train_prim_types_1 = int(self.args.ansi_c_tr_prim_1)
                self.ansi_train_prim_types_1 = [ansi_train_prim_types_1]

            ansi_train_prim_types_2 = self.args.ansi_c_tr_prim_2.split(",")
            self.ansi_train_prim_types_2 = [int(tpt) for tpt in ansi_train_prim_types_2]

            ansi_train_prim_types_3 = self.args.ansi_c_tr_prim_3.split(",")
            self.ansi_train_prim_types_3 = [int(tpt) for tpt in ansi_train_prim_types_3]

            if self.test_performance:
                self.ansi_train_prim_types_1 = self.ansi_train_prim_types_1 + self.ansi_train_prim_types_2 + self.ansi_train_prim_types_3

            ansi_test_prim_types = self.args.ansi_test_prim_types.split(",")
            self.ansi_test_prim_types = [int(tpt) for tpt in ansi_test_prim_types]

            self.train_set_1 = ANSIDataset(
                root=self.dataset_root, filename=self.train_dataset, opt=self.args, csv_path="train_models.csv",
                skip=1, fold=1,
                prim_types=self.ansi_train_prim_types_1, split_type=self.split_type, split_train_test=False,
                train_test_split="train", noisy=False,
                first_n=-1, fixed_order=False
            )

            self.train_set_2 = ANSIDataset(
                root=self.dataset_root, filename=self.train_dataset, opt=self.args, csv_path="train_models.csv",
                skip=1, fold=1,
                prim_types=self.ansi_train_prim_types_2, split_type=self.split_type, split_train_test=False,
                train_test_split="train", noisy=False,
                first_n=-1, fixed_order=False
            )

            self.train_set_3 = ANSIDataset(
                root=self.dataset_root, filename=self.train_dataset, opt=self.args, csv_path="train_models.csv",
                skip=1, fold=1,
                prim_types=self.ansi_train_prim_types_3, split_type=self.split_type, split_train_test=False,
                train_test_split="train", noisy=False,
                first_n=-1, fixed_order=False
            )

            if self.test_performance:
                self.test_set = ANSIDataset(
                    root=self.dataset_root, filename=self.test_dataset, opt=self.args, csv_path="test_models.csv",
                    skip=1, fold=1,
                    prim_types=self.ansi_test_prim_types, split_type=self.split_type, split_train_test=False,
                    train_test_split="test", noisy=False,
                    first_n=-1, fixed_order=False
                )
        ''' Set datasets '''

        ''' Set data samplers '''
        self.train_sampler_1 = torch.utils.data.distributed.DistributedSampler(
            self.train_set_1, num_replicas=hvd.size(), rank=hvd.rank())

        self.train_sampler_2 = torch.utils.data.distributed.DistributedSampler(
            self.train_set_2, num_replicas=hvd.size(), rank=hvd.rank())

        self.train_sampler_3 = torch.utils.data.distributed.DistributedSampler(
            self.train_set_3, num_replicas=hvd.size(), rank=hvd.rank())

        if self.test_performance:
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_set, num_replicas=hvd.size(), rank=hvd.rank())
        ''' Set data samplers '''

        ''' Set dataloaders '''
        self.train_loader_1 = data.DataLoader(
            self.train_set_1, batch_size=self.batch_size,
            sampler=self.train_sampler_1, **kwargs)

        self.train_loader_2 = data.DataLoader(
            self.train_set_2, batch_size=self.batch_size,
            sampler=self.train_sampler_2, **kwargs)

        self.train_loader_3 = data.DataLoader(
            self.train_set_3, batch_size=self.batch_size,
            sampler=self.train_sampler_3, **kwargs)

        if self.test_performance:
            self.test_loader = data.DataLoader(
                self.test_set, batch_size=self.batch_size if not self.args.inference else 1,
                sampler=self.test_sampler, **kwargs)
        ''' Set dataloaders '''

        ''' Set optimizers '''
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

        self.optimizer_C = torch.optim.Adam(
            self.model_C.parameters(),
            lr=self.init_lr * lr_scaler,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        self.optimizer_C = hvd.DistributedOptimizer(
            self.optimizer_C,
            named_parameters=self.model_C.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        hvd.broadcast_parameters(self.model_B.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer_B, root_rank=0)
        hvd.broadcast_parameters(self.model_C.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer_C, root_rank=0)
        ''' Set optimizers '''

        ''' Set working dirs '''
        self.model_dir = "task_{}_stage_{}_nn_inter_loss_{}_inst_part_seg_mixing_type_{}_init_lr_{}_bsz_{}_drop_50_lr_schedule_projection_more_bn_resume_{}".format(
            self.args.task,
            str(self.stage),
            str(self.nn_inter_loss),
            self.landmark_type,
            str(self.init_lr),
            str(batch_size),
            str(True if len(resume) > 0 else False)
        )
        self.model_dir_B = self.model_dir + "_B"
        self.model_dir_C = self.model_dir + "_C"

        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if not os.path.exists("./prm_cache"):
                os.mkdir("./prm_cache")
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir))
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir_B)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir_B))
            if not os.path.exists(os.path.join("./prm_cache", self.model_dir_C)):
                os.mkdir(os.path.join("./prm_cache", self.model_dir_C))
        self.model_dir = "./prm_cache/" + self.model_dir
        self.model_dir_B = "./prm_cache/" + self.model_dir_B
        self.model_dir_C = "./prm_cache/" + self.model_dir_C
        ''' Set working dirs '''

        ''' Set loss model '''
        self.loss_model_save_path = os.path.join(self.model_dir, "loss_model")
        self.loss_model_save_path_B = os.path.join(self.model_dir_B, "loss_model")
        self.loss_model_save_path_C = os.path.join(self.model_dir_C, "loss_model")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            if not os.path.exists(self.loss_model_save_path):
                os.mkdir(self.loss_model_save_path)
            if not os.path.exists(self.loss_model_save_path_B):
                os.mkdir(self.loss_model_save_path_B)
            if not os.path.exists(self.loss_model_save_path_C):
                os.mkdir(self.loss_model_save_path_C)
        self.loss_model = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                  lr_scaler=lr_scaler, init_lr=float(args.init_lr),
                                                  weight_decay=float(args.weight_decay),
                                                  loss_model_save_path=self.loss_model_save_path,
                                                  in_rep_dim=self.map_feat_dim if not args.use_spfn else 128,
                                                  nn_inter_loss=args.nn_inter_loss,
                                                  args=args
                                                  )

        self.loss_model_B = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                    lr_scaler=lr_scaler, init_lr=float(args.init_lr),
                                                    weight_decay=float(args.weight_decay),
                                                    loss_model_save_path=self.loss_model_save_path_B,
                                                    in_rep_dim=self.map_feat_dim if not args.use_spfn else 128,
                                                    nn_inter_loss=args.nn_inter_loss,
                                                    args=args
                                                    )

        self.loss_model_C = ComputingGraphLossModel(pos_dim=3, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03,
                                                    lr_scaler=lr_scaler, init_lr=float(args.init_lr),
                                                    weight_decay=float(args.weight_decay),
                                                    loss_model_save_path=self.loss_model_save_path_C,
                                                    in_rep_dim=self.map_feat_dim if not args.use_spfn else 128,
                                                    nn_inter_loss=args.nn_inter_loss,
                                                    args=args
                                                    )
        self.loss_model.cuda()
        self.loss_model_B.cuda()
        self.loss_model_C.cuda()
        ''' Set loss model '''

        ''' Set optimizer for loss model '''
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

        cur_optimizer_B = torch.optim.Adam(
            self.loss_model_B.parameters(),
            lr=self.init_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)
        self.head_optimizer_B = cur_optimizer_B
        # DISTRIBUTE self.head_optimizer
        self.head_optimizer_B = hvd.DistributedOptimizer(
            cur_optimizer_B,
            named_parameters=self.loss_model_B.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        hvd.broadcast_optimizer_state(self.head_optimizer_B, root_rank=0)
        hvd.broadcast_parameters(self.loss_model_B.state_dict(), root_rank=0)

        cur_optimizer_C = torch.optim.Adam(
            self.loss_model_C.parameters(),
            lr=self.init_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)
        self.head_optimizer_C = cur_optimizer_C
        # DISTRIBUTE self.head_optimizer
        self.head_optimizer_C = hvd.DistributedOptimizer(
            cur_optimizer_C,
            named_parameters=self.loss_model_C.named_parameters(),
            op=hvd.Average,
            gradient_predivide_factor=1.0
        )

        #### BROADCAST the model's state_dict and the optimizer's state_dict ####
        hvd.broadcast_optimizer_state(self.head_optimizer_C, root_rank=0)
        hvd.broadcast_parameters(self.loss_model_C.state_dict(), root_rank=0)
        ''' Set optimizer for loss model '''

        ''' Set related sampling distributions '''
        # number of grouping operations, binary operations and unary operations
        self.nn_grp_opers = args.nn_grp_opers
        self.nn_binary_opers = args.nn_binary_opers
        self.nn_unary_opers = args.nn_unary_opers
        self.nn_in_feats = args.nn_in_feats
        ''' Set related sampling distributions '''

        ''' Set sampling tree '''
        if self.args.v2_tree:
            sampling_tree = DistributionTreeNodeV2
        else:
            sampling_tree = DistributionTreeNode

        if self.args.debug or self.test_performance:
            self.sampling_tree_rt = None
        else:
            self.sampling_tree_rt = sampling_tree(cur_depth=0, nn_grp_opers=self.nn_grp_opers,
                                                  nn_binary_opers=self.nn_binary_opers, nn_unary_opers=self.nn_unary_opers,
                                                  nn_in_feat=self.nn_in_feats, args=args,
                                                  )
        ''' Set related sampling distributions '''

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

    ''' Sample intermediate loss generation dictionaries '''
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

    def metric_average_dict(self, val, name, max_label_nn=10):
        cur_tsr = torch.zeros((max_label_nn,), dtype=torch.float32, )
        for i in val:
            cur_tsr[i] = val[i]
        avg_tsr = hvd.allreduce(cur_tsr, name=name)
        avg_dict = {}
        return avg_tsr

    def _train_one_epoch(
            self, epoch,
            conv_select_types=[0, 0, 0, 0, 0],
            loss_selection_dict={},
            cur_model=None,
            cur_loss_model=None,
            cur_optimizer=None,
            cur_head_optimizer=None,
            cur_loaders=None,
            cur_samplers=None
        ):

        cur_model = self.model if cur_model is None else cur_model
        cur_model.train()
        cur_loss_model = self.loss_model if cur_loss_model is None else cur_loss_model
        cur_loss_model.train()
        cur_optimizer = self.optimizer if cur_optimizer is None else cur_optimizer
        cur_head_optimizer = self.head_optimizer if cur_head_optimizer is None else cur_head_optimizer

        loss_list = []
        loss_nn = []

        cur_loaders = [self.train_loader_1] if cur_loaders is None else cur_loaders
        cur_samplers = [self.train_sampler_1] if cur_samplers is None else cur_samplers

        train_bars = [tqdm(cur_loader) for cur_loader in cur_loaders]
        for cur_sampler in cur_samplers:
            cur_sampler.set_epoch(epoch)

        iouvalue = []

        gt_loss = []

        feat_losses = []
        push_losses = []
        pull_losses = []
        normal_losses = []
        type_losses = []

        tot_labels_to_pull_losses = {}
        tot_labels_to_pull_losses_nn = {}

        for train_bar in train_bars:
            for batch_dicts in train_bar:
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
                    batch_primitives = batch_dicts['T_gt']
                    batch_parameters = batch_dicts['T_param']

                if batch_pos.size(0) == 1:
                    continue

                batch_pos = batch_pos.float().cuda()
                batch_normals = batch_normals.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                batch_primitives = batch_primitives.long().cuda()
                if not self.use_ansi:
                    batch_parameters = batch_parameters.float().cuda()

                batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
                bz, N = batch_pos.size(0), batch_pos.size(1)

                feats = {'normals': batch_normals}
                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model
                )

                seg_pred, gt_l, pred_conf, statistics, fps_idx = rt_values

                x = statistics['x']
                # x = statistics['predicted_feat']
                pred_type = statistics['type_per_point']
                normal_per_point = statistics['normal_per_point']
                feat_loss, pull_loss, push_loss, labels_to_pull_losses, labels_to_pull_losses_nn = compute_embedding_loss(x, batch_inst_seg, batch_primitives)
                normal_loss = compute_normal_loss(normal_per_point, batch_normals)
                type_loss = compute_nnl_loss(pred_type, batch_primitives)

                for cur_label in labels_to_pull_losses:
                    if cur_label not in tot_labels_to_pull_losses:
                        tot_labels_to_pull_losses[cur_label] = labels_to_pull_losses[cur_label]
                        tot_labels_to_pull_losses_nn[cur_label] = labels_to_pull_losses_nn[cur_label]
                    else:
                        tot_labels_to_pull_losses[cur_label] += labels_to_pull_losses[cur_label]
                        tot_labels_to_pull_losses_nn[cur_label] += labels_to_pull_losses_nn[cur_label]

                # loss = feat_loss + normal_loss + type_loss + gt_l

                if self.test_performance:
                    if self.stage == 1:
                        loss = 0.1 * gt_l + feat_loss + normal_loss + type_loss
                    else:
                        loss = feat_loss + normal_loss + type_loss
                else:
                    loss = 0.1 * gt_l + feat_loss

                feat_losses.append(feat_loss.detach().cpu().item() * bz)
                push_losses.append(push_loss.detach().cpu().item() * bz)
                pull_losses.append(pull_loss.detach().cpu().item() * bz)
                normal_losses.append(normal_loss.detach().cpu().item() * bz)
                type_losses.append(type_loss.detach().cpu().item() * bz)

                gt_loss.append(gt_l.detach().cpu().item() * bz)

                self.head_optimizer.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                # self.loss_model.head_optimizer.step()
                self.head_optimizer.step()

                loss_list += [loss.detach().cpu().item() * bz]
                loss_nn.append(bz)
                try:
                    iouvalue.append(-0.0)
                except:
                    iouvalue.append(-0.0)

                train_bar.set_description(
                    'Train Epoch: [{}/{}] feat_loss:{:.3f} push_loss:{:.3f} pull_loss:{:.3f} normal_loss:{:.3f} type_loss:{:.3f} GT_L:{:.3f}'.format(
                        epoch + 1, 200, float(sum(feat_losses) / sum(loss_nn)),
                        float(sum(push_losses) / sum(loss_nn)),
                        float(sum(pull_losses) / sum(loss_nn)),
                        float(sum(normal_losses)) / float(sum(loss_nn)),
                        float(sum(type_losses) / sum(loss_nn)),
                        float(sum(gt_loss) / sum(loss_nn)),
                    ))

        avg_feat_loss = float(sum(feat_losses)) / float(sum(loss_nn))
        avg_push_loss = float(sum(push_losses)) / float(sum(loss_nn))
        avg_pull_loss = float(sum(pull_losses)) / float(sum(loss_nn))
        avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
        avg_normal_loss = float(sum(normal_losses)) / float(sum(loss_nn))
        avg_type_loss = float(sum(type_losses)) / float(sum(loss_nn))
        avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
        avg_feat_loss = self.metric_average(avg_feat_loss, 'tr_feat_loss')
        avg_push_loss = self.metric_average(avg_push_loss, 'tr_push_loss')
        avg_pull_loss = self.metric_average(avg_pull_loss, 'tr_pull_loss')
        avg_loss = self.metric_average(avg_loss, 'tr_tot_loss')
        avg_normal_loss = self.metric_average(avg_normal_loss, 'tr_normal_loss')
        avg_type_loss = self.metric_average(avg_type_loss, 'tr_type_loss')
        avg_gt_loss = self.metric_average(avg_gt_loss, 'tr_gt_loss')

        for cur_label in tot_labels_to_pull_losses:
            tot_labels_to_pull_losses[cur_label] /= tot_labels_to_pull_losses_nn[cur_label]
        avg_label_to_pull_loss = self.metric_average_dict(tot_labels_to_pull_losses, 'tr_ty_pull_losses', 10)

        if hvd.rank() == 0:
            with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                wf.write("Train Epoch: {:d}, feat_loss: {:.4f} push_loss:{:.3f} pull_loss:{:.3f} normal_loss: {:.4f} type_loss: {:.4f} GT_L: {:.4f} ty_pull_losses:{}".format(
                    epoch + 1, avg_feat_loss, avg_push_loss, avg_pull_loss,
                    avg_normal_loss,
                    avg_type_loss,
                    avg_gt_loss,
                    str(avg_label_to_pull_loss)
                ) + "\n")
                wf.close()

            logging.info("Train Epoch: {:d}, feat_loss: {:.4f} push_loss:{:.3f} pull_loss:{:.3f} normal_loss: {:.4f} type_loss: {:.4f} GT_L: {:.4f} ty_pull_losses:{}".format(
                epoch + 1, avg_feat_loss, avg_push_loss, avg_pull_loss,
                avg_normal_loss,
                avg_type_loss,
                avg_gt_loss,
                str(avg_label_to_pull_loss)
            ))
        return avg_feat_loss, avg_pull_loss

    def _test(
            self, epoch, desc="val",
            conv_select_types=[0, 0, 0, 0, 0],
            loss_selection_dict=[],
            cur_model=None,
            cur_loss_model=None,
            cur_loader=None
        ):

        cur_model = self.model if cur_model is None else cur_model
        cur_model.eval()
        cur_loss_model = self.loss_model if cur_loss_model is None else cur_loss_model
        cur_loss_model.eval()

        cur_loader = self.train_loader_1 if cur_loader is None else cur_loader

        with torch.no_grad():

            loss_list = []
            loss_nn = []
            test_bar = tqdm(cur_loader)
            iouvalue = []

            gt_loss = []

            avg_recall = []

            feat_losses = [] # feat_loss?
            push_losses = [] # push_loss?
            pull_losses = [] # pull_loss?
            normal_losses = []
            type_losses = []
            all_miou = []

            tot_labels_to_pull_losses = {}
            tot_labels_to_pull_losses_nn = {}

            for batch_dicts in test_bar:

                if self.use_ansi:
                    batch_pos = batch_dicts['P']
                    batch_normals = batch_dicts['normal_gt']
                    batch_inst_seg = batch_dicts['I_gt']
                    T_gt = batch_dicts['T_gt']
                    # T_gt.size = bz x n_max_instance
                    batch_primitives = batched_index_select(values=T_gt, indices=batch_inst_seg, dim=1)
                else:
                    batch_pos = batch_dicts['gt_pc']
                    batch_normals = batch_dicts['gt_normal']
                    batch_inst_seg = batch_dicts['I_gt_clean']

                if batch_pos.size(0) == 1:
                    continue

                batch_pos = batch_pos.float().cuda()
                batch_normals = batch_normals.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                batch_primitives = batch_primitives.long().cuda()

                batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)

                bz, N = batch_pos.size(0), batch_pos.size(1)
                feats = {'normals': batch_normals}
                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model
                )

                seg_pred, gt_l, pred_conf, statistics, fps_idx = rt_values

                x = statistics['x']
                # x = statistics['predicted_feat']
                pred_type = statistics['type_per_point']
                normal_per_point = statistics['normal_per_point']
                # compute embedding loss?
                feat_loss, pull_loss, push_loss, labels_to_pull_losses, labels_to_pull_losses_nn = \
                    compute_embedding_loss(x, batch_inst_seg, batch_primitives)
                normal_loss = compute_normal_loss(normal_per_point, batch_normals)
                type_loss = compute_nnl_loss(pred_type, batch_primitives)

                for cur_label in labels_to_pull_losses:
                    if cur_label not in tot_labels_to_pull_losses:
                        tot_labels_to_pull_losses[cur_label] = labels_to_pull_losses[cur_label]
                        tot_labels_to_pull_losses_nn[cur_label] = labels_to_pull_losses_nn[cur_label]
                    else:
                        tot_labels_to_pull_losses[cur_label] += labels_to_pull_losses[cur_label]
                        tot_labels_to_pull_losses_nn[cur_label] += labels_to_pull_losses_nn[cur_label]

                if self.test_performance:
                    if self.stage == 1:
                        loss = gt_l + feat_loss + normal_loss + type_loss
                    else:
                        loss = feat_loss + normal_loss + type_loss
                else:
                    loss = gt_l + feat_loss

                feat_losses.append(feat_loss.detach().cpu().item() * bz)
                push_losses.append(push_loss.detach().cpu().item() * bz)
                pull_losses.append(pull_loss.detach().cpu().item() * bz)
                normal_losses.append(normal_loss.detach().cpu().item() * bz)
                type_losses.append(type_loss.detach().cpu().item() * bz)

                gt_loss.append(gt_l.detach().cpu().item() * bz)

                loss_list += [loss.detach().cpu().item() * bz]
                loss_nn.append(bz)
                try:
                    iouvalue.append(-0.0)
                except:
                    iouvalue.append(-0.0)

                test_bar.set_description(
                    'Test Epoch: [{}/{}] feat_loss:{:.3f} push_loss:{:.3f} pull_loss:{:.3f} normal_loss:{:.3f} type_loss:{:.3f} GT_L:{:.3f}'.format(
                    epoch + 1, 200, float(sum(feat_losses) / sum(loss_nn)),
                        float(sum(push_losses) / sum(loss_nn)),
                        float(sum(pull_losses) / sum(loss_nn)),
                    float(sum(normal_losses)) / float(sum(loss_nn)),
                    float(sum(type_losses) / sum(loss_nn)),
                    float(sum(gt_loss) / sum(loss_nn)),
                ))

            avg_feat_loss = float(sum(feat_losses)) / float(sum(loss_nn))
            avg_push_loss = float(sum(push_losses)) / float(sum(loss_nn))
            avg_pull_loss = float(sum(pull_losses)) / float(sum(loss_nn))
            avg_loss = float(sum(loss_list)) / float(sum(loss_nn))
            avg_normal_loss = float(sum(normal_losses)) / float(sum(loss_nn))
            avg_type_loss = float(sum(type_losses)) / float(sum(loss_nn))
            avg_gt_loss = float(sum(gt_loss)) / float(sum(loss_nn))
            avg_feat_loss = self.metric_average(avg_feat_loss, 'feat_loss')
            avg_push_loss = self.metric_average(avg_push_loss, 'push_loss')
            avg_pull_loss = self.metric_average(avg_pull_loss, 'pull_loss')
            avg_loss = self.metric_average(avg_loss, 'tot_loss')
            avg_normal_loss = self.metric_average(avg_normal_loss, 'normal_loss')
            avg_type_loss = self.metric_average(avg_type_loss, 'type_loss')
            avg_gt_loss = self.metric_average(avg_gt_loss, 'gt_loss')
            # avg_miou = self.metric_average(avg_miou, 'miou')

            for cur_label in tot_labels_to_pull_losses:
                tot_labels_to_pull_losses[cur_label] /= tot_labels_to_pull_losses_nn[cur_label]
            avg_label_to_pull_loss = self.metric_average_dict(tot_labels_to_pull_losses, 'tr_ty_pull_losses', 10)

            if hvd.rank() == 0:
                with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                    wf.write("Test_{} Epoch: {:d}, feat_loss: {:.4f} push_loss:{:.4f} pull_loss:{:.4f} normal_loss: {:.4f} type_loss: {:.4f} GT_L: {:.4f} ty_pull_losses:{}".format(    desc,
                    epoch + 1, avg_feat_loss, avg_push_loss, avg_pull_loss,
                    avg_normal_loss,
                    avg_type_loss,
                    avg_gt_loss,
                    str(avg_label_to_pull_loss)
                    # avg_miou,
                ) + "\n")

                logging.info("Test_{} Epoch: {:d}, feat_loss: {:.4f} push_loss:{:.4f} pull_loss:{:.4f} normal_loss: {:.4f} type_loss: {:.4f} GT_L: {:.4f} ty_pull_losses:{}".format(
                    desc,
                    epoch + 1, avg_feat_loss, avg_push_loss, avg_pull_loss,
                    avg_normal_loss,
                    avg_type_loss,
                    avg_gt_loss,
                    str(avg_label_to_pull_loss)
                    # avg_miou
                ))
            return avg_feat_loss, avg_pull_loss

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
            best_model_test_acc = 0.0 # best model test acc
            best_model_idx = 0
            rewards = []
            sampled_loss_dicts = []
            for i_model in range(n_models_per_eps): #
                # cur_selected_loss_dict = self.sample_intermediate_representation_generation()
                ''' Sample and broadcast loss dict list '''
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
                ''' Sample and broadcast loss dict list '''

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
                    self.model.cuda()

                    self.model_B.load_state_dict(
                        torch.load(os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"),
                                   map_location='cpu'
                                   )
                    )
                    self.model_B.cuda()

                    self.model_C.load_state_dict(
                        torch.load(os.path.join(self.model_dir_C, "REIN_init_saved_model.pth"),
                                   map_location='cpu'
                                   )
                    )
                    self.model_C.cuda()
                    ''' LOAD model '''

                    ''' LOAD loss models '''
                    # no i_model parameter is passed to the function
                    self.loss_model.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                           init_lr=self.init_lr,
                                                                           weight_decay=self.weight_decay)
                    self.loss_model.cuda()

                    self.loss_model_B.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                             init_lr=self.init_lr,
                                                                             weight_decay=self.weight_decay)
                    self.loss_model_B.cuda()

                    self.loss_model_C.load_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list,
                                                                             init_lr=self.init_lr,
                                                                             weight_decay=self.weight_decay)
                    self.loss_model_C.cuda()
                    ''' LOAD loss models '''
                    best_test_acc = 0.0

                    for i_iter in range(eps_training_epochs):
                        train_feat_loss, train_pull_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model,
                            cur_loss_model=self.loss_model,
                            cur_loaders=[self.train_loader_1, self.train_loader_2],
                            cur_samplers=[self.train_sampler_1, self.train_sampler_2],
                            cur_optimizer=self.optimizer,
                            cur_head_optimizer=self.head_optimizer
                        )

                        val_feat_loss, val_pull_loss = self._test(
                            base_epoch + i_iter + 1, desc="val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model,
                            cur_loader=self.train_loader_3,
                            cur_loss_model=self.loss_model
                            # r=cur_model_sampled_r
                        )

                        best_test_acc += train_feat_loss - val_feat_loss

                    for i_iter in range(eps_training_epochs):
                        train_feat_loss, train_pull_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_B,
                            cur_loss_model=self.loss_model_B,
                            cur_loaders=[self.train_loader_1, self.train_loader_3],
                            cur_samplers=[self.train_sampler_1, self.train_sampler_3],
                            cur_optimizer=self.optimizer_B,
                            cur_head_optimizer=self.head_optimizer_B
                        )

                        val_feat_loss, val_pull_loss = self._test(
                            base_epoch + i_iter + 1, desc="val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_B,
                            cur_loader=self.train_loader_2,
                            cur_loss_model=self.loss_model_B
                        )

                        best_test_acc += train_feat_loss - val_feat_loss

                    for i_iter in range(eps_training_epochs):
                        train_feat_loss, train_pull_loss = self._train_one_epoch(
                            base_epoch + i_iter + 1,
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_C,
                            cur_loss_model=self.loss_model_C,
                            cur_loaders=[self.train_loader_2, self.train_loader_3],
                            cur_samplers=[self.train_sampler_2, self.train_sampler_3],
                            cur_optimizer=self.optimizer_C,
                            cur_head_optimizer=self.head_optimizer_C
                        )

                        val_feat_loss, val_pull_loss = self._test(
                            base_epoch + i_iter + 1, desc="val",
                            conv_select_types=base_arch_select_type,
                            loss_selection_dict=cur_selected_loss_dict_list,
                            cur_model=self.model_C,
                            cur_loader=self.train_loader_1,
                            cur_loss_model=self.loss_model_C
                        )

                        best_test_acc += train_feat_loss - val_feat_loss

                    best_test_acc /= 3.0
                    rewards.append(best_test_acc)

                    if hvd.rank() == 0:
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.model_dir, f"REIN_saved_model_{i_model}.pth"))
                        self.loss_model.save_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list, i_model)

                        torch.save(self.model_B.state_dict(),
                                   os.path.join(self.model_dir_B, f"REIN_saved_model_{i_model}.pth"))
                        # i_model parameter is passed to the function for clearer saving
                        self.loss_model_B.save_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list, i_model)

                        torch.save(self.model_C.state_dict(),
                                   os.path.join(self.model_dir_C, f"REIN_saved_model_{i_model}.pth"))
                        # i_model parameter is passed to the function for clearer saving
                        self.loss_model_C.save_head_optimizer_by_operation_dicts(cur_selected_loss_dict_list, i_model)

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
                print("Distribution parameters after update: ")
                self.print_dist_params_to_file()
                # Load weights
                self.loss_model.load_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx],
                                                                       init_lr=self.init_lr,
                                                                       weight_decay=self.weight_decay,
                                                                       model_idx=best_model_idx)
                # Save weights
                self.loss_model.save_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx])

                self.loss_model_B.load_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx],
                                                                         init_lr=self.init_lr,
                                                                         weight_decay=self.weight_decay,
                                                                         model_idx=best_model_idx)
                # Save weights
                self.loss_model_B.save_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx])

                self.loss_model_C.load_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx],
                                                                         init_lr=self.init_lr,
                                                                         weight_decay=self.weight_decay,
                                                                         model_idx=best_model_idx)
                # Save weights
                self.loss_model_C.save_head_optimizer_by_operation_dicts(sampled_loss_dicts[best_model_idx])

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

            self.model_C.load_state_dict(
                torch.load(os.path.join(self.model_dir_C, f"REIN_saved_model_{best_model_idx}.pth"),
                           map_location="cpu"
                           ))
            self.model_C.cuda()
            ''' LOAD best model '''

            ''' SAVE as init_model '''
            if hvd.rank() == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))
                torch.save(self.model_B.state_dict(), os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"))
                torch.save(self.model_C.state_dict(), os.path.join(self.model_dir_C, "REIN_init_saved_model.pth"))
            best_test_acc = 0.0
            base_epoch += eps_training_epochs

            ''' GET baseline loss dict '''
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

            self.loss_model_B.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                     weight_decay=self.weight_decay)
            self.loss_model_B.cuda()

            self.loss_model_C.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                     weight_decay=self.weight_decay)
            self.loss_model_C.cuda()

            for i_iter in range(eps_training_epochs):
                train_feat_loss, train_pull_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loss_model=self.loss_model,
                    cur_loaders=[self.train_loader_1, self.train_loader_2],
                    cur_samplers=[self.train_sampler_1, self.train_sampler_2],
                    cur_optimizer=self.optimizer,
                    cur_head_optimizer=self.head_optimizer
                )

                val_feat_loss, val_pull_loss = self._test(
                    base_epoch + i_iter + 1, desc="val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loader=self.train_loader_3,
                    cur_loss_model=self.loss_model
                )

                best_test_acc += train_feat_loss - val_feat_loss

            for i_iter in range(eps_training_epochs):
                train_feat_loss, train_pull_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_B,
                    cur_loss_model=self.loss_model_B,
                    cur_loaders=[self.train_loader_1, self.train_loader_3],
                    cur_samplers=[self.train_sampler_1, self.train_sampler_3],
                    cur_optimizer=self.optimizer_B,
                    cur_head_optimizer=self.head_optimizer_B
                )

                val_feat_loss, val_pull_loss = self._test(
                    base_epoch + i_iter + 1, desc="val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_B,
                    cur_loader=self.train_loader_2,
                    cur_loss_model=self.loss_model_B
                )
                best_test_acc += train_feat_loss - val_feat_loss

            for i_iter in range(eps_training_epochs):
                train_feat_loss, train_pull_loss = self._train_one_epoch(
                    base_epoch + i_iter + 1,
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_C,
                    cur_loss_model=self.loss_model_C,
                    cur_loaders=[self.train_loader_2, self.train_loader_3],
                    cur_samplers=[self.train_sampler_2, self.train_sampler_3],
                    cur_optimizer=self.optimizer_C,
                    cur_head_optimizer=self.head_optimizer_C
                )

                val_feat_loss, val_pull_loss = self._test(
                    base_epoch + i_iter + 1, desc="val",
                    conv_select_types=base_arch_select_type,
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model_C,
                    cur_loader=self.train_loader_1,
                    cur_loss_model=self.loss_model_C
                )
                best_test_acc += train_feat_loss - val_feat_loss

            baseline = best_test_acc / 3.0
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

    def beam_searh_for_best(self, base_dict=[], max_nn=3, keep_nn=2, conv_select_types=[0, 0, 0]):
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
            self.loss_model.load_head_optimizer_by_operation_dicts(cur_feed_dicts, init_lr=self.init_lr,
                                                                   weight_decay=self.weight_decay)
            self.loss_model.cuda()

            self.loss_model_B.load_head_optimizer_by_operation_dicts(cur_feed_dicts, init_lr=self.init_lr,
                                                                     weight_decay=self.weight_decay)
            self.loss_model_B.cuda()

            self.loss_model_C.load_head_optimizer_by_operation_dicts(cur_feed_dicts, init_lr=self.init_lr,
                                                                     weight_decay=self.weight_decay)
            self.loss_model_C.cuda()

            #### LOAD model's weights ####
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, "REIN_init_saved_model.pth"), map_location="cpu")
            )
            self.model_B.load_state_dict(
                torch.load(os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"), map_location="cpu")
            )
            self.model_C.load_state_dict(
                torch.load(os.path.join(self.model_dir_C, "REIN_init_saved_model.pth"), map_location="cpu")
            )
            cur_maxx_val_acc = -999.0

            for i in range(val_training_epochs):
                train_feat_loss, train_pull_loss = self._train_one_epoch(
                    i + 1,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model,
                    cur_loss_model=self.loss_model,
                    cur_loaders=[self.train_loader_1, self.train_loader_2],
                    cur_samplers=[self.train_sampler_1, self.train_sampler_2],
                    cur_optimizer=self.optimizer,
                    cur_head_optimizer=self.head_optimizer
                )
                # return

                val_feat_loss, val_pull_loss = self._test(
                    i + 1, desc="partnet_val",
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model,
                    cur_loader=self.train_loader_3,
                    cur_loss_model=self.loss_model
                )

                train_feat_loss2, train_pull_loss2 = self._train_one_epoch(
                    i + 1,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts, desc="train",
                    cur_model=self.model_B,
                    cur_loss_model=self.loss_model_B,
                    cur_loaders=[self.train_loader_1, self.train_loader_3],
                    cur_samplers=[self.train_sampler_1, self.train_sampler_3],
                    cur_optimizer=self.optimizer_B,
                    cur_head_optimizer=self.head_optimizer_B
                )

                val_feat_loss2, val_pull_loss2 = self._test(
                    i + 1, desc="partnet_val",
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model_B,
                    cur_loader=self.train_loader_2,
                    cur_loss_model=self.loss_model_B
                )

                train_feat_loss3, train_pull_loss3 = self._train_one_epoch(
                    i + 1,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts, desc="train",
                    cur_model=self.model_C,
                    cur_loss_model=self.loss_model_C,
                    cur_loaders=[self.train_loader_2, self.train_loader_3],
                    cur_samplers=[self.train_sampler_2, self.train_sampler_3],
                    cur_optimizer=self.optimizer_C,
                    cur_head_optimizer=self.head_optimizer_C
                )

                val_feat_loss3, val_pull_loss3 = self._test(
                    i + 1, desc="partnet_val",
                    conv_select_types=conv_select_types,
                    loss_selection_dict=cur_feed_dicts,
                    cur_model=self.model_C,
                    cur_loader=self.train_loader_1,
                    cur_loss_model=self.loss_model_C
                )

                cur_maxx_val_acc = min(cur_maxx_val_acc, val_feat_loss + val_feat_loss2 + val_feat_loss3)

            val_accs.append((j, cur_maxx_val_acc))
        #### sort validation accs ####
        val_accs = sorted(val_accs, key=lambda kk: kk[1], reverse=True)
        #### GET topk dicts ####
        topk_dicts = [binary_dicts[kk[0]] for kki, kk in enumerate(val_accs[:keep_nn])]
        tot_dicts_ordered = [binary_dicts[kk[0]] for kki, kk in enumerate(val_accs)]

        if hvd.rank() == 0:
            with open(os.path.join(self.model_dir, "selected_dicts.txt"), "a") as wf:
                wf.write(f"{1 + 1}:\n")
                for cur_rt_dict, cur_val_acc_item in zip(tot_dicts_ordered, val_accs):
                    cur_val_acc = cur_val_acc_item[1]
                    wf.write(f"{str(base_dict + cur_rt_dict)}\t{cur_val_acc}\n")
                wf.close()
        return topk_dicts

    def train_all(self):
        # eps_training_epochs = 10
        eps_training_epochs = 1
        n_models_per_eps = 4
        tot_eps = 30
        ''' SAVE current model weights as initial weights '''
        if hvd.rank() == 0:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "REIN_init_saved_model.pth"))
            torch.save(self.model_B.state_dict(), os.path.join(self.model_dir_B, "REIN_init_saved_model.pth"))
            torch.save(self.model_C.state_dict(), os.path.join(self.model_dir_C, "REIN_init_saved_model.pth"))
        baseline_loss_dict = []

        if self.args.beam_search:
            baseline_value = torch.tensor([1, 1, 0], dtype=torch.long)
            rt_dicts = self.beam_searh_for_best(base_dict=[], max_nn=3, keep_nn=2, conv_select_types=baseline_value.tolist())
            print(rt_dicts)
            return

        best_test_acc = 0.0

        baseline_value = torch.tensor([0, 0, 0], dtype=torch.long)



        not_improved_num_epochs = 0

        if self.test_performance:
            eps_training_epochs = 100
            # baseline_loss_dict = [{'gop': 2, 'uop': 1, 'bop': 16, 'lft_chd': {'uop': 2, 'oper': 3}, 'rgt_chd': {'gop': 0, 'uop': 3, 'bop': 1, 'lft_chd': {'uop': 4, 'oper': 3}, 'rgt_chd': {'uop': 5, 'oper': 2}}}]
            baseline_loss_dict = [{'gop': 2, 'uop': 0, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 1}, 'rgt_chd': {'uop': 5, 'oper': 1}}]

            ''' LOAD loss model (head & optimizer) '''
            # LOAD heads for features prediction
            self.loss_model.load_head_optimizer_by_operation_dicts(baseline_loss_dict, init_lr=self.init_lr,
                                                                   weight_decay=self.weight_decay)
            self.loss_model.cuda()
            ''' LOAD loss model (head & optimizer) '''

            best_feat_loss = 1000.0
            best_test_feat_loss = 0.0
            test_pull_loss = 0.0
            for i_iter in range(eps_training_epochs):
                train_feat_loss, train_pull_loss = self._train_one_epoch(
                    i_iter + 1,
                    conv_select_types=baseline_value.tolist(),
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loss_model=self.loss_model,
                    cur_loaders=[self.train_loader_1],
                    cur_samplers=[self.train_sampler_1],
                    cur_optimizer=self.optimizer,
                    cur_head_optimizer=self.head_optimizer
                )

                val_feat_loss, val_pull_loss = self._test(
                    i_iter + 1, desc="val",
                    conv_select_types=baseline_value.tolist(),
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loader=self.train_loader_3,
                    cur_loss_model=self.loss_model
                )

                test_feat_loss, test_pull_loss = self._test(
                    i_iter + 1, desc="test",
                    conv_select_types=baseline_value.tolist(),
                    loss_selection_dict=baseline_loss_dict,
                    cur_model=self.model,
                    cur_loader=self.test_loader, # test loader
                    cur_loss_model=self.loss_model
                )

                if val_feat_loss < best_feat_loss:
                    best_feat_loss = val_feat_loss
                    best_test_feat_loss = test_feat_loss
                    not_improved_num_epochs = 0
                    if hvd.rank() == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "Loss_best_saved_model.pth"))
                else:
                    not_improved_num_epochs += 1
                    if not_improved_num_epochs >= 20:
                        self.adjust_learning_rate_by_factor(0.7)
                        not_improved_num_epochs = 0
                        print("Decrease the learning rate by 0.7...")
                        with open(os.path.join(self.model_dir, "logs.txt"), "a") as wf:
                            wf.write("Decrease the learning rate by 0.7...\n")
                            wf.close()
                best_test_acc += train_feat_loss - val_feat_loss
            print(f"Best feat loss: {best_feat_loss}, best_test_feat_loss: {best_test_feat_loss}")
            exit(0)

        for i_iter in range(eps_training_epochs):
            mem = psutil.virtual_memory()
            print(f"Before {i_iter}-th iter: mem.total = {mem.total}, mem.available = {mem.available}, mem.free = {mem.free}")
            swap_meme = psutil.swap_memory()
            print(f"Before {i_iter}-th iter: swap.total = {swap_meme.total}, swap.used = {swap_meme.used}, swap.free = {swap_meme.free}")

            train_feat_loss, train_pull_loss = self._train_one_epoch(
                i_iter + 1,
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model,
                cur_loss_model=self.loss_model,
                cur_loaders=[self.train_loader_1, self.train_loader_2],
                cur_samplers=[self.train_sampler_1, self.train_sampler_2],
                cur_optimizer=self.optimizer,
                cur_head_optimizer=self.head_optimizer
            )

            val_feat_loss, val_pull_loss = self._test(
                i_iter + 1, desc="val",
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model,
                cur_loader=self.train_loader_3,
                cur_loss_model=self.loss_model
            )

            mem = psutil.virtual_memory()
            print(
                f"After {i_iter}-th iter: mem.total = {mem.total}, mem.available = {mem.available}, mem.free = {mem.free}")
            swap_meme = psutil.swap_memory()
            print(
                f"After {i_iter}-th iter: swap.total = {swap_meme.total}, swap.used = {swap_meme.used}, swap.free = {swap_meme.free}")

            best_test_acc += train_feat_loss - val_feat_loss

        for i_iter in range(eps_training_epochs):
            train_feat_loss, train_pull_loss = self._train_one_epoch(
                i_iter + 1,
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model_B,
                cur_loss_model=self.loss_model_B,
                cur_loaders=[self.train_loader_1, self.train_loader_3],
                cur_samplers=[self.train_sampler_1, self.train_sampler_3],
                cur_optimizer=self.optimizer_B,
                cur_head_optimizer=self.head_optimizer_B
            )

            val_feat_loss, val_pull_loss = self._test(
                i_iter + 1, desc="val",
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model_B,
                cur_loader=self.train_loader_2,
                cur_loss_model=self.loss_model_B
            )

            best_test_acc += train_feat_loss - val_feat_loss

        for i_iter in range(eps_training_epochs):
            train_feat_loss, train_pull_loss = self._train_one_epoch(
                i_iter + 1,
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model_C,
                cur_loss_model=self.loss_model_C,
                cur_loaders=[self.train_loader_2, self.train_loader_3],
                cur_samplers=[self.train_sampler_2, self.train_sampler_3],
                cur_optimizer=self.optimizer_C,
                cur_head_optimizer=self.head_optimizer_C
            )

            val_feat_loss, val_pull_loss = self._test(
                i_iter + 1, desc="val",
                conv_select_types=baseline_value.tolist(),
                loss_selection_dict=baseline_loss_dict,
                cur_model=self.model_C,
                cur_loader=self.train_loader_1,
                cur_loss_model=self.loss_model_C
            )

            best_test_acc += train_feat_loss - val_feat_loss


        baseline = best_test_acc / 3.0

        base_epoch = 0
        if hvd.rank() == 0:
            logging.info(f"Baseline = {baseline}")

        each_search_epochs = self.sea_interval
        base_model_select_arch_types = baseline_value.tolist()

        for i_eps in range(tot_eps):

            baseline, baseline_loss_dict = self.rein_train_search_loss(
                base_epoch=base_epoch, base_arch_select_type=base_model_select_arch_types, baseline=baseline
            )

            base_epoch += each_search_epochs
