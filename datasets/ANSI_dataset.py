# code adapted from https://github.com/lingxiaoli94/SPFN/blob/master/spfn/lib/dataset.py

import sys
import os

import re
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
import h5py
import random
import pandas
from collections import Counter
from .data_util import rotate_perturbation_point_cloud, jitter_point_cloud, \
    shift_point_cloud, random_scale_point_cloud, rotate_point_cloud, get_dataset_root, create_primitive_from_dict, \
NAME_TO_ID_DICT, extract_parameter_data_as_dict

from model.utils import farthest_point_sampling, get_knn_idx, batched_index_select

EPS = np.finfo(np.float32).eps


def create_unit_data_from_hdf5(f, n_max_instances, noisy, fixed_order=False, check_only=False, shuffle=True):
    '''
        f will be a h5py group-like object
    '''

    P = f['noisy_points'][()] if noisy else f['gt_points'][()]  # Nx3
    normal_gt = f['gt_normals'][()]  # Nx3
    I_gt = f['gt_labels'][()]  # N

    # for j in range(I_gt.shape[0]):
    #     if int(I_gt[j].item()) == -1:
    #         print("-1!")
    #         break

    P_gt = []

    # next check if soup_ids are consecutive
    found_soup_ids = []
    soup_id_to_key = {}
    soup_prog = re.compile('(.*)_soup_([0-9]+)$')
    for key in list(f.keys()):
        m = soup_prog.match(key)
        if m is not None:
            soup_id = int(m.group(2))
            found_soup_ids.append(soup_id)  # soup idx?
            soup_id_to_key[soup_id] = key  #
    found_soup_ids.sort()
    n_instances = len(found_soup_ids)
    if n_instances == 0:
        return None
    for i in range(n_instances):
        if i not in found_soup_ids:
            print('{} is not found in soup ids!'.format(i))
            return None

    instances = []
    for i in range(n_instances):
        g = f[soup_id_to_key[i]]
        P_gt_cur = g['gt_points'][()]
        P_gt.append(P_gt_cur)
        meta = pickle.loads(g.attrs['meta'])
        primitive = create_primitive_from_dict(meta)
        if primitive is None:
            return None
        instances.append(primitive)

    if n_instances > n_max_instances:
        print('n_instances {} > n_max_instances {}'.format(n_instances, n_max_instances))
        return None

    if np.amax(I_gt) >= n_instances:
        print('max label {} > n_instances {}'.format(np.amax(I_gt), n_instances))
        return None

    if check_only:
        return True

    T_gt = [NAME_TO_ID_DICT[primitive['type']] for primitive in instances]
    # SET the primitive type of the backgroud  part to `-1`
    T_gt = T_gt + [-1] + [-1 for _ in range(n_max_instances - n_instances)]
    # T_gt.extend([0 for _ in range(n_max_instances - n_instances + 1)])  # K # primitive type

    n_total_points = P.shape[0]
    n_gt_points_per_instance = P_gt[0].shape[0]
    P_gt.extend(
        [np.zeros(dtype=float, shape=[n_gt_points_per_instance, 3]) for _ in range(n_max_instances - n_instances + 1)])

    # convert everything to numpy array
    P_gt = np.array(P_gt)
    T_gt = np.array(T_gt)

    # SET the background points to `n_instances`-th segment
    I_gt[I_gt == -1] = n_instances


    if shuffle:
        # shuffle per point information around
        perm = np.random.permutation(n_total_points)
        P = P[perm]
        normal_gt = normal_gt[perm]
        I_gt = I_gt[perm]

    result = {
        'P': P,
        'normal_gt': normal_gt,
        'P_gt': P_gt,  #
        'I_gt': I_gt,  # N
        'T_gt': T_gt,  # K --- type
        'curnmask': np.array([n_instances], dtype=np.long)
    }

    # Next put in primitive parameters
    for fitter_cls in NAME_TO_ID_DICT:
        result.update(extract_parameter_data_as_dict(instances, n_max_instances + 1, type=fitter_cls))

    return result


class ANSIDataset(data.Dataset):
    def __init__(
            self, root, filename, opt, csv_path, skip=1, fold=1,
            prim_types=[], split_type=False, split_train_test=False, train_test_split="train", noisy=False, first_n=-1, fixed_order=False):

        ''' CREATE dataset root & load data '''
        self.root = get_dataset_root(root, file_name=csv_path)
        # self.data_path = open(os.path.join(self.root, filename), 'r')
        self.opt = opt
        self.fixed_order = fixed_order
        self.first_n = first_n
        self.noisy = noisy
        self.split_type = self.opt.split_type if self.opt is not None else True
        self.n_max_instances = self.opt.n_max_instances if self.opt is not None else 30
        self.split_train_test = split_train_test
        self.inference = self.opt.inference if "inference" in self.opt is not None else False
        # point cloud augmentation
        self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]

        if not self.split_type:
            self.csv_raw = pandas.read_csv(os.path.join(self.root, csv_path), delimiter=',', header=None)
            self.hdf5_file_list = list(self.csv_raw[0])
            csv_folder = os.path.dirname(csv_path)
            self.hdf5_file_list = [os.path.join(csv_folder, p) for p in self.hdf5_file_list if not os.path.isabs(p)]

            if not fixed_order:
                random.shuffle(self.hdf5_file_list)
            if first_n != -1: # clamp the file list
                self.hdf5_file_list = self.hdf5_file_list[:first_n]

            self.len = len(self.hdf5_file_list)

            self.basename_list = [os.path.splitext(os.path.basename(p))[0] for p in self.hdf5_file_list]
            self.n_data = len(self.hdf5_file_list)
            self.first_iteration_finished = False
        else:
            self.n_clusters = self.opt.ansi_n_clusters
            prim_types = [int(item) for item in prim_types if item != ',']
            self.data_list = []
            for prm_idx in prim_types:
                prm_file_name = f"clustered_{self.n_clusters}_primitive_{prm_idx}_data_names.txt"
                if self.inference:
                    prm_file_name = f"clustered_{self.n_clusters}_primitive_{prm_idx}_data_names_inference.txt"
                elif self.split_train_test and train_test_split != "test":
                    prm_file_name = f"clustered_{self.n_clusters}_primitive_{prm_idx}_data_names_{train_test_split}.txt"

                prm_data_file_path = open(os.path.join(self.root, prm_file_name), "r")
                prm_data_list = [item.strip() for item in prm_data_file_path.readlines()]
                self.data_list += prm_data_list
            self.tru_len = len(self.data_list)
            self.len = self.tru_len
            self.n_data = len(self.data_list)
            self.first_iteration_finished = False
            self.hdf5_file_list = self.data_list


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

        path = self.hdf5_file_list[index]
        # name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(os.path.join(self.root, path), 'r') as handle:
            data = create_unit_data_from_hdf5(handle, self.n_max_instances, noisy=self.noisy,
                                              fixed_order=self.fixed_order, shuffle=not self.fixed_order)
            assert data is not None  # assume data are all clean

        return data

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
