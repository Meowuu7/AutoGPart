import torch
import torch.nn as nn
from model.pointnetpp_segmodel_sea import PointNetPPInstSeg
from model.primitive_fitting_net_sea import PrimitiveFittingNet
# from model.pointnetpp_segmodel_cls_sea import InstSegNet
# from model.pointnetpp_segmodel_cls_sea_v2 import InstSegNet
from model.motion_segmodel_cls_sea_v2 import InstSegNet

# from datasets.Indoor3DSeg_dataset import Indoor3DSemSeg
from datasets.instseg_dataset import InstSegmentationDataset
from datasets.ABC_dataset import ABCDataset
from datasets.ANSI_dataset import ANSIDataset
from datasets.motionseg_dataset import PartSegmentationMetaInfoDataset
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
from .trainer_utils import get_masks_for_seg_labels, compute_param_loss, DistributionTreeNode, DistributionTreeNodeV2, DistributionTreeNodeArch
from datasets.partnet_dataset import PartNetInsSeg
from model.loss_utils import get_one_hot
from model.model_util import set_bn_not_training, set_grad_to_none
from model.loss_utils import compute_miou, npy
from model.utils import mean_shift
from model.loss_utils import compute_embedding_loss
from model.abc_utils import compute_entropy, construction_affinity_matrix_normal


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
        ''' SET horovod configurations '''
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(5)
        kwargs = {'num_workers': 5, 'pin_memory': True}

        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        self.kwargs = kwargs
        ''' SET horovod configurations '''

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
        self.nn_inter_loss = self.args.nn_inter_loss

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
        self.conv_select_types = [0,0,1] # [0,0,0]
        self.point_feat_selection = int(self.args.point_feat_selection)
        self.point_geo_feat_selection = int(self.args.point_geo_feat_selection)
        # print(self.args.contrast_selection)
        contrast_selection = self.args.contrast_selection.split(",")
        self.contrast_selection = [int(cs) for cs in contrast_selection]

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

        # with FileLock(os.path.expanduser("~/.horovod_lock")):
        if not os.path.exists("./prm_cache"):
            os.mkdir("./prm_cache")
        if not os.path.exists(os.path.join("./prm_cache", self.model_dir)):
            os.mkdir(os.path.join("./prm_cache", self.model_dir))
        self.model_dir = "./prm_cache/" + self.model_dir
        ''' SET working dirs '''

        ''' SET working dir for loss model '''
        self.loss_model_save_path = os.path.join(self.model_dir, "loss_model")
        # with FileLock(os.path.expanduser("~/.horovod_lock")):
        if os.path.exists(self.loss_model_save_path):
            print(f"=== REMOVE the existing loss model file from {self.loss_model_save_path} ===")
            import shutil
            shutil.rmtree(self.loss_model_save_path)

        if not os.path.exists(self.loss_model_save_path):
            os.mkdir(self.loss_model_save_path)
        self.args.loss_model_save_path = self.loss_model_save_path
        ''' SET working dir for loss model '''

        ''' GET model '''
        self.args.lr_scaler = 1.0
        self.model = InstSegNet(
            n_layers=self.n_layers,
            feat_dims=self.feat_dims,
            n_samples=self.n_samples,
            map_feat_dim=self.map_feat_dim,
            args=self.args  # args
        )
        self.model.cuda()
        # self.dataparallel = dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1
        ''' GET model '''

        ''' SET datasets & data-loaders & data-samplers '''
        # DATASET ROOT & NUMBER OF MASKS
        self.dataset_root = self.args.dataset_root
        self.nmasks = int(self.args.pred_nmasks)

        # DATASETS for train & val & test
        self.train_dataset = self.args.train_dataset
        self.val_dataset = self.args.val_dataset
        self.test_dataset = self.args.test_dataset

        # DATA-SPLITTING configurations
        self.split_type = self.args.split_type
        self.split_train_test = self.args.split_train_test

        ### GET test data-types for all categories in test setting ####
        self.pure_test = self.args.pure_test
        pure_test_types = self.args.pure_test_types.split(";")
        self.pure_test_types = [str(tpt) for tpt in pure_test_types]
        ### GET test data-types for all categories in test setting ####

        # TRAIN & VAL & TEST PartNet types
        partnet_train_types = self.args.partnet_train_types.split(";")
        self.partnet_train_types = [str(tpt) for tpt in partnet_train_types]
        partnet_val_types = self.args.partnet_val_types.split(";")
        self.partnet_val_types = [str(tpt) for tpt in partnet_val_types]
        partnet_test_types = self.args.partnet_test_types.split(";")
        self.partnet_test_types = [str(tpt) for tpt in partnet_test_types]

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

        self.test_set = PartSegmentationMetaInfoDataset(
            root=self.dataset_root, split='real_test', npoints=512, nmask=10, relrot=True,
            load_file_name="tot_part_motion_meta_info.npy",
            shape_types=["03642806", "03636649", "02691156", "03001627", "02773838", "02954340", "03467517",
                         "03790512",
                         "04099429", "04225987", "03624134", "02958343", "03797390", "03948459", "03261776",
                         "04379243"],
            real_test=True, args=args
        )
        print("Loaded...")

        self.test_loader = data.DataLoader(
            self.test_set, batch_size=self.batch_size,
            # sampler=self.test_sampler,
            shuffle=False,
            **kwargs)
        ''' SET datasets & data-loaders & data-samplers '''

        ''' SET optimizers '''
        lr_scaler = 1.0
        self.lr_scaler = lr_scaler

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.init_lr * lr_scaler,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        ''' SET optimizers '''

        ''' SET related sampling distributions '''
        # number of grouping operations, binary operations and unary operations
        self.nn_grp_opers = args.nn_grp_opers  # 4  # bz x N x k x f... -> bz x N x f...: sum, mean, max, svd
        self.nn_binary_opers = args.nn_binary_opers  # 6  # add, minus, element-wise multiply, cross product, cartesian product, matrix-vector product
        self.nn_unary_opers = args.nn_unary_opers  # 7  # identity, square, 2, -, -2, inv, orth
        self.nn_in_feats = args.nn_in_feats  # 2


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
            ''' SET loss model '''

    def get_gt_conf(self, momask):
        if momask.size(1) > momask.size(2):
            momask = momask.transpose(1, 2)
        gt_conf = torch.sum(momask, 2)
        gt_conf = torch.where(gt_conf > 0, torch.ones_like(gt_conf), torch.zeros_like(gt_conf)).float()
        return momask, gt_conf

    def get_nn_segmentations(self, batch_inst_seg):
        bz = batch_inst_seg.size(0)
        tot_nn_segs = 0
        for i in range(bz):
            cur_inst_seg = batch_inst_seg[i]
            cur_seg_nn = int(torch.max(cur_inst_seg).item()) + 1
            tot_nn_segs += cur_seg_nn
        return tot_nn_segs

    def _clustering_test(
        self, epoch, desc="val",
        conv_select_types=[0, 0, 0, 0, 0],
        loss_selection_dict=[],
        save_samples=False,
        sample_interval=20,
        inference_prim_types=[2,6]
    ):

        save_stats_path = os.path.join(self.model_dir, "inference_saved")
        self.args.inference = False

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
                batch_pos = batch_dicts['pc1_af_glb']
                batch_flow = batch_dicts['flow12']
                batch_momasks = batch_dicts['motion_seg_masks']

                batch_inst_seg = torch.argmax(batch_momasks, dim=-1)  # assume momask.size = bz x N x nmasks

                cur_batch_nn_seg = self.get_nn_segmentations(batch_inst_seg)

                # if batch_pos.size(0) == 1:
                #     continue
                if batch_pos.size(-1) > 3:
                    batch_pos = batch_pos[:, :, :3]

                batch_pos = batch_pos.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                # batch_momasks = get_one_hot(batch_inst_seg, 200)
                batch_momasks = batch_momasks.float().cuda()
                batch_flow = batch_flow.float().cuda()

                # batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
                bz, N = batch_pos.size(0), batch_pos.size(1)

                feats = {'flow': batch_flow}

                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model if not self.in_model_loss_model else self.model.intermediate_loss if self.args.add_intermediat_loss else None
                )

                seg_pred, gt_l, pred_conf, losses, fps_idx = rt_values
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
                totx = losses['x']

                for jj in range(totx.size(0)):
                    x = totx[jj].unsqueeze(0)
                    cur_batch_inst_seg = batch_inst_seg[jj].unsqueeze(0)
                    feat_loss, pull_loss, push_loss, _, _ = compute_embedding_loss(x, cur_batch_inst_seg)
                    feat_losses.append(feat_loss.detach().cpu().item() * 1)

                    loss_nn.append(1)
                    # print(x.size())

                    feat_ent = feat_ent_weight - float(npy(compute_entropy(x)))

                    weight_ent.append(feat_ent)
                    spec_embedding_list.append(x)

                    weighted_list = []
                    # norm_weight_ent = weight_ent / np.linalg.norm(weight_ent)
                    for i in range(len(spec_embedding_list)):
                        weighted_list.append(spec_embedding_list[i] * weight_ent[i])

                    spectral_embedding = torch.cat(weighted_list, dim=-1)

                    spec_cluster_pred = mean_shift(spectral_embedding, bandwidth=bandwidth)

                    miou = compute_miou(spec_cluster_pred, cur_batch_inst_seg)

                    all_miou.append(miou.detach().item())

                test_bar.set_description(
                    'Test_{} Epoch: [{}/{}] Iou:{:.3f} feat_loss:{:.3f} normal_loss:{:.3f} type_loss:{:.3f}'.format(
                        desc,
                        epoch + 1, 200, float(sum(all_miou) / len(all_miou)),
                        float(sum(feat_losses) / sum(loss_nn)),
                        float(sum(normal_losses)) / float(sum(loss_nn)),
                        float(sum(type_losses) / sum(loss_nn)),
                    ))

            with open(os.path.join(self.model_dir, "test_logs.txt"), "a") as wf:
                wf.write('Iou:{:.3f} feat_loss:{:.3f} '.format(
                        float(sum(all_miou) / len(all_miou)),
                        float(sum(feat_losses) / sum(loss_nn)),) + "\n")
                wf.close()
            return float(sum(all_miou) / len(all_miou))

    def _test(
        self, epoch, desc="val",
        conv_select_types=[0, 0, 0],
        loss_selection_dict=[],
        cur_loader=None
    ):

        conv_select_types = self.conv_select_types
        self.model.eval()

        if not self.in_model_loss_model:
            self.loss_model.eval()

        with torch.no_grad():

            totn = 0

            if cur_loader is None:
                if desc == "val":
                    cur_loader = self.val_loader
                elif desc == "test":
                    cur_loader = self.test_loader
                else:
                    raise ValueError(f"Unrecognized desc: {desc}")

            loss_list = []
            loss_nn = []
            test_bar = tqdm(cur_loader)
            iouvalue = []

            gt_loss = []

            avg_recall = []
            tot_seg_nns = []

            batch_seg_pred = []

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
                    batch_pos = batch_pos[:, :, :3]

                batch_pos = batch_pos.float().cuda()
                batch_inst_seg = batch_inst_seg.long().cuda()
                # batch_momasks = get_one_hot(batch_inst_seg, 200)
                batch_momasks = batch_momasks.float().cuda()
                batch_flow = batch_flow.float().cuda()

                # batch_momasks = get_masks_for_seg_labels(batch_inst_seg, maxx_label=self.n_max_instances)
                bz, N = batch_pos.size(0), batch_pos.size(1)

                feats = {'flow': batch_flow}

                rt_values = self.model(
                    pos=batch_pos, masks=batch_momasks, inst_seg=batch_inst_seg, feats=feats,
                    conv_select_types=conv_select_types,
                    loss_selection_dict=loss_selection_dict,
                    loss_model=self.loss_model if not self.in_model_loss_model else self.model.intermediate_loss if self.args.add_intermediat_loss else None
                )

                seg_pred, gt_l, pred_conf, losses, fps_idx = rt_values

                if "iou" in losses:
                    iou_value = losses["iou"]
                    cur_avg_recall = 0.0
                    seg_loss = None
                else:
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

                    seg_loss = None

                avg_recall.append(cur_avg_recall * bz)

                inst_seg_pred = torch.argmax(seg_pred, dim=1)
                batch_seg_pred.append(inst_seg_pred)

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

            batch_seg_pred = torch.cat(batch_seg_pred, dim=0)
            np.save(os.path.join("curr_motion_test_pred.npy"), batch_seg_pred.detach().cpu().numpy())

            # if hvd.rank() == 0:
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

    def train_all(self):

        assert self.args.resume != ""

        ''' LOAD dicts safely '''
        print(f"Loading weights from {self.args.resume}.")
        # self.args.resume should be the file path to pre-laod weights #
        ori_dict = torch.load(os.path.join(self.args.resume), map_location='cpu')
        part_dict = dict()
        model_dict = self.model.state_dict()
        for k in ori_dict:
            if k in model_dict:
                v = ori_dict[k]
                part_dict[k] = v
        model_dict.update(part_dict)
        self.model.load_state_dict(model_dict)
        ''' LOAD dicts safely '''

        ''' TEST performance '''
        # test_acc, test_gt_loss = self._test(
        #     0 + 1, desc="test",
        #     conv_select_types=[0,0,0],
        #     loss_selection_dict=[]
        #     # r=cur_model_sampled_r
        # )

        test_acc = self._clustering_test(epoch=1)

        print(f"Test iou = {test_acc}.")
        ''' TEST performance '''
