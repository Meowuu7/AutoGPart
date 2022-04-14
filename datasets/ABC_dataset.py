# code adapted from https://github.com/SimingYan/HPNet/blob/main/dataloader/ABCDataset.py

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
import h5py
import random
from collections import Counter
from .data_util import rotate_perturbation_point_cloud, jitter_point_cloud, \
    shift_point_cloud, random_scale_point_cloud, rotate_point_cloud, get_dataset_root, create_primitive_from_dict

from model.utils import farthest_point_sampling, get_knn_idx, batched_index_select

EPS = np.finfo(np.float32).eps


class ABCDataset(data.Dataset):
    def __init__(
            self, root, filename, opt, skip=1, fold=1,
            prim_types=[], split_type=False, split_train_test=False, train_test_split="train"):

        #### dataset root & load data ####
        self.root = get_dataset_root(root)
        self.data_path = open(os.path.join(self.root, filename), 'r')
        self.opt = opt
        # point cloud augmentation
        self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]

        if 'train' in filename:
            self.augment = self.opt.augment
            self.if_normal_noise = self.opt.if_normal_noise
        else:
            self.augment = 0
            self.if_normal_noise = 0

        self.split_type = self.opt.split_type
        self.split_train_test = self.opt.split_train_test
        self.prim_types = prim_types
        if not self.split_type:
            self.data_list = [item.strip() for item in self.data_path.readlines()]
            self.skip = skip

            self.data_list = self.data_list[::self.skip]
            self.tru_len = len(self.data_list)
            self.len = self.tru_len * fold  # len?
        else:
            self.data_list = []
            # todo: true len and len? --- what does `fold` use for?
            # prim_types = eval(prim_types)
            # print(type(prim_types), prim_types)
            prim_types = [int(item) for item in prim_types if item != ',']
            print(type(prim_types), prim_types)
            n_clusters = 15
            for prm_idx in prim_types:
                # prm_data_file_name = f"primitive_{prm_idx}_data.txt"
                prm_data_file_name = f"clustered_{n_clusters}_primitive_{prm_idx}_data_names.txt"
                if split_train_test:
                    prm_data_file_name = train_test_split + "_" + prm_data_file_name
                prm_data_file_path = open(os.path.join(self.root, prm_data_file_name), "r")
                prm_data_list = [item.strip() for item in prm_data_file_path.readlines()]
                self.data_list += prm_data_list
                self.tru_len = len(self.data_list)
                self.len = self.tru_len

    def reindex_labels(self, labels):
        labels_to_nn = {}
        for i in range(labels.shape[0]):
            cur_lab = int(labels[i].item())
            if cur_lab not in labels_to_nn:
                labels_to_nn[cur_lab] = 1
            else:
                labels_to_nn[cur_lab] += 1
        sorted_labels_nn = sorted(labels_to_nn.items(), key=lambda i: i[1], reverse=True)
        old_label_to_new_label = {item[0]: i for i, item in enumerate(sorted_labels_nn)}
        for i in range(labels.shape[0]):
            cur_lab = int(labels[i].item())
            new_lab = old_label_to_new_label[cur_lab]
            labels[i] = new_lab
        return labels

    def __getitem__(self, index):

        ret_dict = {}  # return dictionary
        index = index % self.tru_len

        data_file = os.path.join(self.root, self.data_list[index] + '.h5')

        with h5py.File(data_file, 'r') as hf:
            points = np.array(hf.get("points"))
            labels = np.array(hf.get("labels"))  # each point's label
            normals = np.array(hf.get("normals"))
            primitives = np.array(hf.get("prim"))
            primitive_param = np.array(hf.get("T_param"))

        # n_sampling = 2048
        # # fps_idx = farthest_point_sampling(pos=torch.from_numpy(points[:, :3][None, :, :]).float(), n_sampling=5000)
        # permidx = np.random.permutation(points.shape[0])[:n_sampling]
        # # fps_idx = fps_idx.detach().cpu().numpy()
        # points = points[permidx]
        # labels = labels[permidx]
        # normals = normals[permidx]
        # primitives = primitives[permidx]
        # primitive_param = primitive_param[permidx]

        # point cloud data augmentation
        if self.augment:
            points = self.augment_routines[np.random.choice(np.arange(5))](points[None, :, :])[0]

        # add noise to normal
        if self.if_normal_noise:
            noise = normals * np.clip(
                np.random.randn(points.shape[0], 1) * 0.01,
                a_min=-0.01,
                a_max=0.01)
            points = points + noise.astype(np.float32)

        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = normals
        ret_dict['T_gt'] = primitives.astype(int)
        ret_dict['T_param'] = primitive_param

        print(primitive_param.size(), points.size(), labels.size())


        # set small number primitive as background
        counter = Counter(labels)
        mapper = np.ones([labels.max() + 1]) * -1
        # k --- label idx
        keys = [k for k, v in counter.items() if v > 100]
        keys = sorted(keys, key=lambda i: counter[i], reverse=True)

        # maximum_key = keys[0]
        # for i in range()

        # keys --- as indexes
        # len(keys) ---
        if len(keys):
            mapper[keys] = np.arange(
                len(keys))  # reindex label idxes --- valid labels are re-indexed and invalid labels are set to -1
        label = mapper[labels]  # label vector stores valid labels of each point in the shape
        # index the array
        # I_gt --- I_gt --- labels --- labeldic
        ret_dict['I_gt'] = label.astype(int)
        # get clean primitives
        clean_primitives = np.ones_like(primitives) * -1
        valid_mask = label != -1
        # select clean primitives
        clean_primitives[valid_mask] = primitives[valid_mask]
        ret_dict['T_gt'] = clean_primitives.astype(int)

        ret_dict['index'] = self.data_list[index]  # index --- only the unique index of each object

        small_idx = label == -1
        full_labels = label
        # all idxes ---- to ---- len(keys) - 1
        # labels[small_idx]
        # full_labels[small_idx] = labels[small_idx] + len(keys)
        full_labels[small_idx] = len(keys)
        # ret_dict['I_gt_clean'] = self.reindex_labels(full_labels.astype(int))
        ret_dict['I_gt_clean'] = full_labels.astype(int)
        # return points, normals
        # valid masks --- number of valid masks
        ret_dict['curnmask'] = np.array([len(keys)], dtype=np.long)
        return ret_dict

    def __len__(self):
        return self.len


if __name__ == '__main__':

    abc_dataset = ABCDataset(
        root=
        '/home/ysm/project/2021_SIG_Primitive/Primitive_Detection/thirdparty/parsenet-codebase/data/shapes',
        filename='train_data.h5')

    for idx in range(len(abc_dataset)):
        example = abc_dataset[idx]
        import ipdb

        ipdb.set_trace()
