# Adapted from https://github.com/tiangeluo/Learning-to-Group/blob/master/partnet/data/partnet.py

import os
import os.path as osp
from collections import OrderedDict, defaultdict
import json

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

# from shaper.utils.pc_util import normalize_points as normalize_points_np
# from partnet.utils.torch_pc import group_points, select_points
# from partnet.utils.torch_pc import normalize_points as normalize_points_torch
from IPython import embed

def normalize_points_np(points):
    """Normalize point cloud

    Args:
        points (np.ndarray): (n, 3)

    Returns:
        np.ndarray: normalized points

    """
    assert points.ndim == 2 and points.shape[1] == 3
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    new_points = points / norm
    return new_points


class PartNetInsSeg(Dataset):
    cat_file = './shape_names.txt'

    def __init__(self,
                 root_dir,
                 split,
                 normalize=True, # keep true
                 transform=None,
                 shape='',
                 stage1='',
                 level=-1,
                 cache_mode=True
                 ):
        self.root_dir = root_dir
        self.split = split
        self.normalize = normalize
        self.transform = transform
        self.shape_levels = self._load_cat_file()
        self.cache_mode = cache_mode
        self.folder_list = self._prepare_file_list(shape, level)

        self.cache = defaultdict(list)
        self.meta_data = []
        self._load_data()
        print('{:s}: with {} shapes'.format(
            self.__class__.__name__,  len(self.meta_data)))

    def _load_cat_file(self):
        # Assume that the category file is put under root_dir.
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
        # with open(self.cat_file, 'r') as fid:
            shape_levels = OrderedDict()
            for line in fid:
                shape, levels = line.strip().split('\t')
                levels = tuple([int(l) for l in levels.split(',')])
                max_level = max(levels)
                shape_levels[shape] = max_level # levels
        return shape_levels

    def _prepare_file_list(self, shape, level):
        shape = shape if len(shape) > 0 else None
        level = level if level > 0 else None
        # prepare files according to shape and level, if none, all will be loaded
        folder_list = []
        if (type(shape).__name__ == 'list') == False:
            shape_arr=list()
            shape_arr.append(shape)
        else:
            shape_arr = shape

        for shape in shape_arr:
            level_of_cur_shape = self.shape_levels[shape]
            folder_list.append('{}-{}'.format(shape, level_of_cur_shape))
        return folder_list

    def _load_data(self):
        for folder in self.folder_list:
            folder_path = osp.join(self.root_dir, folder)
            files = os.listdir(folder_path)
            num_list = []
            for f in files:
                num_list.append(int(f.split('-')[-1].split('.')[0]))
            rank = np.argsort(np.array(num_list))
            #for fname in os.listdir(folder_path):
            for k in rank:
                fname = files[k]
                if fname.startswith(self.split) and fname.endswith('h5'):
                    if self.split=='test':
                        folder_path = folder_path.replace('for_detection', 'gt')
                        data_path = osp.join(folder_path, fname)
                        print('loading {}'.format(data_path))
                        with h5py.File(data_path, mode='r') as f:
                            num_samples = f['pts'].shape[0]
                            if self.cache_mode:
                                # point cloud [N, 10000, 3]
                                self.cache['points'].append(f['pts'][:])
                                # instance idx [N, 200, 10000]
                                self.cache['gt_mask'].append(f['gt_mask'][:])
                                # valid class indicator [N, 200]
                                self.cache['gt_valid'].append(f['gt_mask_valid'][:])
                                # valid class indicator [N, 10000]
                                self.cache['gt_other_mask'].append(f['gt_mask_other'][:])
                                gt_valid = f['gt_mask_valid'][:]
                                gt_mask = f['gt_mask'][:]
                                gt_label = f['gt_mask_label'][:]
                                real_gt_label = np.zeros([gt_label.shape[0], 10000]).astype(np.uint8)
                                for i in range(real_gt_label.shape[0]):
                                    for j in range(np.sum(gt_valid[i])):
                                        real_gt_label[i][gt_mask[i,j]] = gt_label[i,j]
                                # semantics class [N, 10000]
                                self.cache['gt_label'].append(real_gt_label)
                    else:
                        data_path = osp.join(folder_path, fname)
                        print('loading {}'.format(data_path))
                        with h5py.File(data_path, mode='r') as f:
                            num_samples = f['pts'].shape[0]
                            if self.cache_mode:
                                # point cloud [N, 10000, 3]
                                self.cache['points'].append(f['pts'][:])
                                # semantics class [N, 10000]
                                self.cache['gt_label'].append(f['gt_label'][:])
                                # instance idx [N, 200, 10000]
                                self.cache['gt_mask'].append(f['gt_mask'][:])
                                # valid class indicator [N, 200]
                                self.cache['gt_valid'].append(f['gt_valid'][:])
                                # valid class indicator [N, 10000]
                                self.cache['gt_other_mask'].append(f['gt_other_mask'][:])
                    for ind in range(num_samples):
                        self.meta_data.append({
                            'offset': ind,
                            'size': num_samples,
                            'path': data_path,
                        })
                    data_path = data_path.replace('.h5', '.json')
                    print('loading {}'.format(data_path))
                    with open(data_path) as f:
                        meta_data_list = json.load(f)
                        #assert num_samples == len(meta_data_list)
                        for ind in range(num_samples):
                            self.meta_data[len(self.meta_data) - num_samples + ind].update(
                                meta_data_list[ind]
                            )

        for k, v in self.cache.items():
            self.cache[k] = np.concatenate(v, axis=0)

    def reindex_shape_seg(self, shape_seg):
        old_seg_to_new_seg = {}
        old_seg_to_nn = {}
        for i in range(shape_seg.shape[0]):
            old_seg = int(shape_seg[i].item())
            if old_seg not in old_seg_to_nn:
                old_seg_to_nn[old_seg] = 1
            else:
                old_seg_to_nn[old_seg] += 1
        srt_items = sorted(old_seg_to_nn.items(), key=lambda ii: ii[1], reverse=True)
        old_seg_to_new_seg = {tt[0]: i for i, tt in enumerate(srt_items)}
        # print(srt_items)
        # print(old_seg_to_new_seg)

        for j in range(shape_seg.shape[0]):
            old_seg = int(shape_seg[j].item())
            new_seg = old_seg_to_new_seg[old_seg]
            shape_seg[j] = new_seg if new_seg < 199 else 199

        return shape_seg

    def __getitem__(self, index):
        if self.cache_mode:
            points = self.cache['points'][index]
            # (10000, )
            gt_label = self.cache['gt_label'][index]
            # (200, 10000)
            gt_mask = self.cache['gt_mask'][index]
            # (200,)
            gt_valid = self.cache['gt_valid'][index]
            # (10000, )
            gt_other_mask = self.cache['gt_other_mask'][index]
        else:
            with h5py.File(self.meta_data[index]['path'], mode='r') as f:
                ind = self.meta_data[index]['offset']
                points = f['pts'][ind]
                gt_label = f['gt_label'][ind]
                gt_mask = f['gt_mask'][ind]
                gt_valid = f['gt_valid'][ind]
                gt_other_mask = f['gt_other_mask'][ind]

        ins_id = self.reindex_shape_seg(gt_label)

        if self.normalize:
            points = normalize_points_np(points)
        if self.transform is not None:
            points, ins_id = self.transform(points, ins_id)

        out_dict = dict(
            points=points,
            ins_id=ins_id,
            gt_mask=np.array(gt_mask,dtype=np.uint8),
            gt_valid=np.array(gt_valid,dtype=np.uint8),
            # meta=self.meta_data[index]
            # gt_label=gt_label,
            # gt_other_mask=gt_other_mask
        )

        return out_dict

    def get(self, anno_id):
        assert isinstance(anno_id, str)
        index = -1
        for idx, meta_data in enumerate(self.meta_data):
            if anno_id == meta_data['anno_id']:
                index = idx
                break
        assert index > 0, '{} not found'.format(anno_id)
        return self[index]

    def __len__(self):
        return len(self.meta_data)