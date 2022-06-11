# Adapted from https://github.com/ericyi/articulated-part-induction/blob/master/data_prep.py


import os
import os.path
import json
import numpy as np
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# import provider
from torch.utils import data
import scipy.io as sio
import copy
from model.utils import farthest_point_sampling
from .data_util import *

''' Dataset for the mobility-based part segmentation task '''
class PartSegmentationMetaInfoDataset(data.Dataset):
    def __init__(
            self, root="./data", npoints=512, split='train', nmask=10, relrot=True, landmark_type="net", buf_lm=False,
            load_file_name="tot_part_motion_meta_info.npy", shape_types=["03642806", "04379243"], split_data=None, real_test=False,
            part_net_seg=False, rt_inter=False, partnet_split=False, flow_est=False, args=None, split_types=False
    ):
        super(PartSegmentationMetaInfoDataset, self).__init__()
        self.dataset_path = os.path.join(root, f"seg_{split}")
        self.masks_names = "momasks"
        if "test" in split:
            self.dataset_path = os.path.join(root, split)
            self.masks_names = "seg1"
        self.buf_lm = buf_lm
        self.nmask = nmask
        self.relrot = relrot
        self.npoints = npoints
        # self.data = sio.loadmat(self.dataset_path)
        self.landmark_type = landmark_type
        self.split = split
        self.real_test = real_test
        self.part_net_seg = part_net_seg
        self.rt_inter = rt_inter
        self.flow_est = flow_est
        self.args = args

        self.lm = dict()
        self.peridx = dict()

        print(f"ROOT_DIR = {ROOT_DIR}")

        if self.split == 'train' or self.split == 'val' or (not self.real_test):

            self.data = {}

            if not self.part_net_seg:
                # if not os.path.exists("/data-input/motion_part_split_meta_info"):
                motion_data_root = os.path.join(root, "data", "motion_part_split_meta_info")
                fl_path = os.path.join(motion_data_root, f"all_type_tot_{self.split}_tot_part_motion_meta_info.npy")
                data = np.load(fl_path, allow_pickle=True).item()
                if split_types:
                    self.data = {}
                    shape_types_dict = {st: 1 for st in shape_types}
                    for k in data:
                        shp_ty_k, shp_idx_k = k.split("_")
                        if shp_ty_k in shape_types_dict:
                            self.data[k] = data[k]
                else:
                    self.data = data
            else:
                self.shp_types_to_number = dict()
                # partnet_root = os.path.join(root, "..", "..")
                partnet_root = root
                for shp_name in shape_types:
                    cur_shp_meta_file_pth = os.path.join(partnet_root, "part_net_meta_info_category", shp_name)
                    if not partnet_split:
                        cur_shp_merged_meta_info = np.load(os.path.join(cur_shp_meta_file_pth, "motion_part_meta_info_merged.npy"), allow_pickle=True).item()
                    else:
                        cur_shp_merged_meta_info = np.load(os.path.join(cur_shp_meta_file_pth, f"{split}_motion_part_meta_info_merged.npy"), allow_pickle=True).item()
                    for shp_idx in cur_shp_merged_meta_info:
                        self.data[shp_idx] = cur_shp_merged_meta_info[shp_idx]
                    self.shp_types_to_number[shp_name] = len(cur_shp_merged_meta_info)

            print(f"{self.split} data loaded with total length = {len(self.data)}")
        else:
            # self.dataset_path = os.path.join(root, "..", f"sf2f_test_est.mat")
            self.dataset_path = os.path.join(root, "data", f"sf2f_test_est.mat")
            self.masks_names = "seg1"
            self.nmask = nmask
            self.relrot = relrot
            self.npoints = npoints
            self.data = sio.loadmat(self.dataset_path)

        self.new_idx_to_old_idx = {}
        self.reindex_data_index()

        if split == 'train':
            self.whether_mov_p = np.array([0.0, 1.0 ], dtype=np.float)
        else:
            self.whether_mov_p = np.array([0.0, 1.0], dtype=np.float)
        self.get_trans_encoding_to_trans_dir()

        self.num_mov_parts_to_cnt = {}

    def get_trans_encoding_to_trans_dir(self):
        trans_dir_to_trans_mode = {
            (0, 1, 2): 1,
            (0, 2): 2,
            (0, 1): 3,
            (1, 2): 4,
            (0,): 5,
            (1,): 6,
            (2,): 7,
            (): 0
        }
        self.trans_mode_to_trans_dir = {trans_dir_to_trans_mode[k]: k for k in trans_dir_to_trans_mode}
        self.base_transition_vec = [
            np.array([1.0, 0.0, 0.0], dtype=np.float),
            np.array([0.0, 1.0, 0.0], dtype=np.float),
            np.array([0.0, 0.0, 1.0], dtype=np.float),
        ]

    def get_test_data(self):
        return self.test_data

    def reindex_data_index(self):
        old_idx_to_new_idx = {}
        ii = 0
        for old_idx in self.data:
            old_idx_to_new_idx[old_idx] = ii
            ii += 1
        self.new_idx_to_old_idx = {old_idx_to_new_idx[k]: k for k in old_idx_to_new_idx}

    def generate_3d(self, smaller=False):
        """Generate a 3D random rotation matrix.
        Returns:
            np.matrix: A 3D rotation matrix.
        """
        x1, x2, x3 = np.random.rand(3)

        rng = 0.25
        offset = 0.15

        if not smaller:
            effi = np.random.uniform(-rng, rng, (1,)).item()
        else:
            rng = 0.15
            offset = 0.05
            effi = np.random.uniform(-rng, rng, (1,)).item()

        if effi < 0:
            effi -= offset
        else:
            effi += offset
        theta = effi * np.pi


        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        w = np.array([np.cos(2 * np.pi * x2) * np.sqrt(x3),
                      np.sin(2 * np.pi * x2) * np.sqrt(x3),
                      np.sqrt(1 - x3)], dtype=np.float)
        w_matrix = np.array(
            [[0, -float(w[2]), float(w[1])], [float(w[2]), 0, -float(w[0])], [-float(w[1]), float(w[0]), 0]]
        )

        rotation_matrix = np.eye(3) + w_matrix * sin_theta + (w_matrix.dot(w_matrix)) * (1. - cos_theta)

        return rotation_matrix

    def reindex_shape_seg(self, shape_seg):
        old_seg_to_new_seg = {}
        ii = 0
        for i in range(shape_seg.shape[0]):
            old_seg = int(shape_seg[i].item())
            if old_seg not in old_seg_to_new_seg:
                old_seg_to_new_seg[old_seg] = ii
                ii += 1
            new_seg = old_seg_to_new_seg[old_seg]
            shape_seg[i] = new_seg
            # old_seg_to_new_seg[old_seg] = ii
            # ii += 1
        return shape_seg

    def transit_pos_by_transit_vec(self, trans_pos):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        # for flow test...
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale * 2
        return trans_pos_af_pos, tdir * 0.1 * trans_scale * 2

    def transit_pos_by_transit_vec_dir(self, trans_pos, tdir):
        # tdir = np.zeros((3,), dtype=np.float)
        # axis_dir = np.random.choice(3, 1).item()
        # tdir[int(axis_dir)] = 1.
        trans_scale = np.random.uniform(0.0, 1.0, (1,)).item()
        trans_pos_af_pos = trans_pos + tdir * 0.1 * trans_scale
        return trans_pos_af_pos, tdir * 0.1 * trans_scale

    def get_random_transition_dir_scale(self):
        tdir = np.random.uniform(-1.0, 1.0, (3,))
        tdir = tdir / (np.sqrt(np.sum(tdir ** 2)).item() + 1e-9)
        trans_scale = np.random.uniform(1.0, 2.0, (1,)).item()
        return tdir, trans_scale

    def decode_trans_dir(self, trans_encoding):
        trans_dir = self.trans_mode_to_trans_dir[trans_encoding]
        return [self.base_transition_vec[d] for ii, d in enumerate(trans_dir)]

    def __getitem__(self, index):

        if self.split == 'train' or (not self.real_test):
            reindex_idx = self.new_idx_to_old_idx[index]
            # cur_shape = copy.deepcopy(self.data[reindex_idx])
            cur_shape = self.data[reindex_idx]
            pc1 = copy.deepcopy(cur_shape['pc1'])
            inst_part_idx_to_part_idx_to_meta_info = cur_shape['part_meta_info']
            chosen_whether_mov_index = np.random.choice(2, 1, p=self.whether_mov_p).item()
            pc2 = copy.deepcopy(pc1)
            if 'sem_seg' in cur_shape:
                shape_seg = cur_shape['sem_seg']
            else:
                shape_seg = cur_shape['inst_seg']
            # print(shape_seg.shape, pc1.shape, cur_shape.keys())
            # motion_seg = np.zeros_like(shape_seg, dtype=np.long)
            motion_seg = np.zeros((pc1.shape[0], ), dtype=np.long)
            chosen_num_moving_parts = 0
            gt_rotation = np.reshape(np.eye(3, dtype=np.float), (1, 3, 3)) * np.ones((pc1.shape[0], 1, 1), dtype=np.float)
            gt_transition = np.zeros((pc1.shape[0], 3, 1), dtype=np.float)
            if chosen_whether_mov_index == 1:
                # mov_inst_part_idx = list(inst_part_idx_to_part_idx_to_meta_info.keys())
                mov_inst_part_idx = [
                    k for k in inst_part_idx_to_part_idx_to_meta_info if ((len(inst_part_idx_to_part_idx_to_meta_info[k]['rot_meta']) > 0
                    or len(inst_part_idx_to_part_idx_to_meta_info[k]['trans_meta']) > 0) and
                    inst_part_idx_to_part_idx_to_meta_info[k]['points_idx'].shape[0] > 30)]
                if len(mov_inst_part_idx) > 0:
                    chosen_num_moving_parts = np.random.choice(len(mov_inst_part_idx), 1).item() + 1
                    chosen_num_moving_parts = min(chosen_num_moving_parts, 4)
                    # chosen_num_moving_parts = min(chosen_num_moving_parts, 1)
                    chosen_mov_inst_new_idx = np.random.choice(len(mov_inst_part_idx), chosen_num_moving_parts, replace=False)

                    cur_mov_pts_number = 0
                    for jj in range(chosen_mov_inst_new_idx.shape[0]):

                        jjjj = int(chosen_mov_inst_new_idx[jj].item())
                        mov_inst = mov_inst_part_idx[jjjj]
                        mov_inst_meta_info = inst_part_idx_to_part_idx_to_meta_info[mov_inst]
                        #
                        mov_points_idx = mov_inst_meta_info['points_idx']
                        rot_meta_info_list = mov_inst_meta_info['rot_meta']
                        trans_meta_info_list = mov_inst_meta_info['trans_meta']

                        if pc1.shape[0] - (cur_mov_pts_number + mov_points_idx.shape[0]) < 30:
                            break

                        if len(rot_meta_info_list) > 0:
                            for_chosen_rot_meta_idx_list = [ii for ii in range(len(rot_meta_info_list)) if
                                                            int(rot_meta_info_list[ii][0].item()) > 0]
                        else:
                            for_chosen_rot_meta_idx_list = []
                        if len(trans_meta_info_list) > 0:
                            for_chosen_trans_meta_idx_list = [ii for ii in range(len(trans_meta_info_list)) if
                                                              int(trans_meta_info_list[ii][0].item()) > 0]
                        else:
                            for_chosen_trans_meta_idx_list = []

                        if len(for_chosen_trans_meta_idx_list) > 0 and len(for_chosen_rot_meta_idx_list) > 0:
                            choose_trans_rot = int(np.random.choice(2, 1, p=[0.7, 0.3]).item())
                        elif len(for_chosen_trans_meta_idx_list) > 0:
                            choose_trans_rot = 1
                        elif len(for_chosen_rot_meta_idx_list) > 0:
                            choose_trans_rot = 0
                        else:
                            choose_trans_rot = 2

                        if choose_trans_rot == 0:
                            chosen_rot_meta_idx = np.random.choice(len(for_chosen_rot_meta_idx_list), 1).item()
                            chosen_rot_meta_idx = for_chosen_rot_meta_idx_list[chosen_rot_meta_idx]

                            rot_meta_info = rot_meta_info_list[chosen_rot_meta_idx]
                            rot_meta_type = int(rot_meta_info[0].item())
                            rot_base_point = rot_meta_info[1:4]
                            possi_rot_axis_vec = decode_rotation_info(rot_meta_type)
                            #
                            if len(possi_rot_axis_vec) > 0:
                                chosen_rot_axis_idx = np.random.choice(len(possi_rot_axis_vec), 1).item()
                                rot_vec = possi_rot_axis_vec[chosen_rot_axis_idx]
                                bf_rotate_pos = pc1[mov_points_idx, :]
                                af_rotate_pos, applied_rotation, applied_transition = rotate_by_vec_pts(rot_vec, rot_base_point, bf_rotate_pos)
                                motion_seg[mov_points_idx] = jj + 1
                                cur_mov_pts_number += mov_points_idx.shape[0]
                                pc2[mov_points_idx, :] = af_rotate_pos
                                # multiplication between rotation matrices
                                gt_rotation[mov_points_idx] = np.matmul(np.reshape(applied_rotation, (1, 3, 3)), gt_rotation[mov_points_idx])
                                # gt_rotation[mov_points_idx, :, :] = np.reshape(applied_rotation, (1, 3, 3)).dot(gt_rotation[mov_points_idx, :, :])
                                gt_transition[mov_points_idx] = np.matmul(np.reshape(applied_rotation, (1, 3, 3)), gt_transition[mov_points_idx] - np.reshape(applied_transition, (1, 3, 1))) + np.reshape(applied_transition, (1, 3, 1))
                                # gt_transition[mov_points_idx, :, 0] = np.reshape(np.reshape(applied_rotation, (1, 3, 3)).dot(np.reshape(gt_transition[mov_points_idx, :, :], (mov_points_idx.shape[0], 3, 1))), (mov_points_idx.shape[0], 3))
                            else:
                                chosen_num_moving_parts -= 1
                                if chosen_num_moving_parts == 0:
                                    chosen_whether_mov_index = 0
                        elif choose_trans_rot == 1:
                            chosen_trans_meta_idx = np.random.choice(len(for_chosen_trans_meta_idx_list), 1).item()
                            chosen_trans_meta_idx = for_chosen_trans_meta_idx_list[chosen_trans_meta_idx]

                            trans_meta_info = trans_meta_info_list[chosen_trans_meta_idx]
                            trans_meta_type = int(trans_meta_info[0].item())
                            possi_trans_axis_vec = self.decode_trans_dir(trans_meta_type)
                            if len(possi_trans_axis_vec) > 0:
                                chosen_trans_axis_idx = np.random.choice(len(possi_trans_axis_vec), 1).item()
                                trans_vec = possi_trans_axis_vec[chosen_trans_axis_idx]
                                bf_transit_pos = pc1[mov_points_idx, :]
                                af_transit_pos, applied_transition = self.transit_pos_by_transit_vec_dir(trans_pos=bf_transit_pos,
                                                                                     tdir=trans_vec)
                                motion_seg[mov_points_idx] = jj + 1
                                cur_mov_pts_number += mov_points_idx.shape[0]
                                pc2[mov_points_idx, :] = af_transit_pos
                                gt_transition[mov_points_idx] += np.reshape(applied_transition, (1, 3, 1))
                            else:
                                chosen_num_moving_parts -= 1
                                if chosen_num_moving_parts == 0:
                                    chosen_whether_mov_index = 0
                        else:
                            chosen_num_moving_parts -= 1
                            if chosen_num_moving_parts == 0:
                                chosen_whether_mov_index = 0
                else:
                    chosen_whether_mov_index = 0

            if chosen_whether_mov_index == 0:
                chosen_num_moving_parts = 0

            pc1_af_rel_trans = pc1

            R1 = self.generate_3d(smaller=True)
            pc1_af_rel = np.transpose(np.matmul(R1, np.transpose(pc1_af_rel_trans, [1, 0])), [1, 0])

            rel_rot_for_pc2 = R1.T
            gt_rotation = np.matmul(gt_rotation, np.reshape(rel_rot_for_pc2, (1, 3, 3)))
            flow12 = pc2 - pc1_af_rel # flows used for training
            rd_num = np.random.choice(2, 1).item()
            pc1_af_glb = pc1_af_rel
            pc2_af_glb = pc2

            gt_transform_vec = np.concatenate(
                [np.reshape(gt_rotation, (pc1.shape[0], 9)), np.reshape(gt_transition, (pc1.shape[0], 3))], axis=-1
            )

            permidx = np.random.permutation(pc1_af_glb.shape[0])[:self.npoints]
            pc1_af_glb = pc1_af_glb[permidx, :]
            gt_transform_vec = gt_transform_vec[permidx, :]
            pc1_af_glb = np.concatenate([pc1_af_glb, gt_transform_vec], axis=-1)
            flow12 = flow12[permidx, :]
            shape_seg = self.reindex_shape_seg(shape_seg)
            # shape_seg = shape_seg[permidx]
            shape_seg_masks = np.eye(self.nmask)[np.minimum(shape_seg, self.nmask - 1)[permidx].astype('int32')]
            # motion_seg = motion_seg[permidx]
            motion_seg_masks = np.eye(self.nmask)[np.minimum(motion_seg, self.nmask - 1)[permidx].astype('int32')]

            permidx2 = np.random.permutation(pc2_af_glb.shape[0])[:self.npoints]
            pc2_af_glb = pc2_af_glb[permidx2, :]

            rt_dict = {
                'whether_mov_index': np.array([chosen_whether_mov_index], dtype=np.long),
                'num_moving_parts': np.array([chosen_num_moving_parts], dtype=np.long),
                'pc1_af_glb': pc1_af_glb, # Pos1 after global transformation
                'pc2_af_glb': pc2_af_glb, # Pos2 after global transformation
                'flow12': flow12, # Flow matrices
                'shape_seg_masks': shape_seg_masks, # Shape segmentation masks
                'motion_seg_masks': motion_seg_masks, # Motion segmentation masks
                'pc1_af_rel': pc1_af_rel[permidx, :], # Pos1 after relative transformation between aprts
                'pc1': pc1[permidx, :] # Original Pos1
            }
        else:
            chosen_num_moving_parts = 1
            pc1 = copy.deepcopy(self.data['pc1'][index])
            flow12 = copy.deepcopy(self.data['flow_est'][index])
            momasks = copy.deepcopy(self.data[self.masks_names][index])
            momasks = momasks.astype('int32') - 1
            momasks = np.eye(self.nmask)[np.minimum(momasks, self.nmask - 1).astype('int32')]
            chosen_whether_mov_index = 1
            shape_seg_masks = np.zeros((pc1.shape[0], self.nmask), dtype=np.float)
            motion_seg_masks = momasks
            rt_dict = {
                'whether_mov_index': np.array([chosen_whether_mov_index], dtype=np.long),
                'num_moving_parts': np.array([chosen_num_moving_parts], dtype=np.long),
                'flow12': flow12,
                'shape_seg_masks': shape_seg_masks,
                'motion_seg_masks': motion_seg_masks,
                'pc1_af_rel': pc1,
                'pc1': pc1,
                'pc1_af_glb': pc1
            }

        return rt_dict

    def __len__(self):
        if self.split == 'train' or (not self.real_test):
            return len(self.data)
        else:
            return self.data['pc1'].shape[0]
        # return self.data['pc1'].shape[0]

    def get_num_moving_parts_to_cnt(self):
        return self.num_mov_parts_to_cnt

    def reset_num_moving_parts_to_cnt(self):
        self.num_mov_parts_to_cnt = {}



if __name__ == '__main__':
    pass
