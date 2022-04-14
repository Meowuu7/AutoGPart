
import numpy as np
import os
import h5py
from sklearn.cluster import KMeans
import pandas
import pickle
# import torch
from collections import Counter


def down_sample_partnet(root_path):
    partnet_file_name = "tot_part_motion_meta_info.npy"
    part_meta_info = np.load(os.path.join(root_path, partnet_file_name), allow_pickle=True).item()
    # pc1; inst_seg
    down_sampled_shps = dict()
    down_num_pts = 512
    down_num_pts = 1024
    ii = 0
    for shp_idx in part_meta_info:
        if ii % 1000 == 0:
            print(ii, shp_idx)
        ii += 1
        cur_meta_info = part_meta_info[shp_idx]
        pos = cur_meta_info['pc1']
        inst_seg = cur_meta_info['inst_seg']
        permidx = np.random.permutation(pos.shape[0])[:down_num_pts]
        sampled_pos = pos[permidx]
        sampled_inst_seg = inst_seg[permidx]
        cur_shp_info = {"pos": sampled_pos, "inst_seg": sampled_inst_seg}
        down_sampled_shps[shp_idx] = cur_shp_info
    np.save(os.path.join(root_path, f"partnet_shapes_{down_num_pts}.npy"), down_sampled_shps)


def get_part_idx_to_num(inst_seg, mmax_nmask=10):
    part_idx_to_nn = dict()
    for i in range(inst_seg.shape[0]):
        cur_part = int(inst_seg[i].item())
        if cur_part not in part_idx_to_nn:
            part_idx_to_nn[cur_part] = 1
        else:
            part_idx_to_nn[cur_part] += 1
    sorted_part_idx_to_nn = sorted(part_idx_to_nn.items(), key=lambda ite: ite[1], reverse=True)
    old_part_idx_to_new_part_idx = {}
    for i, ite in enumerate(sorted_part_idx_to_nn):
        cur_part_idx = ite[0]
        old_part_idx_to_new_part_idx[cur_part_idx] = i if i < mmax_nmask else mmax_nmask - 1
    return old_part_idx_to_new_part_idx

def clip_part_num(root_path):
    down_num_pts = 512
    partnet_file_name = f"tot_part_motion_meta_info"
    shape_info = np.load(os.path.join(root_path, partnet_file_name), allow_pickle=True).item()
    new_shape_info = {}
    mmax_nmask = 10
    for shp_idx in shape_info:
        cur_shape_info = shape_info[shp_idx]
        inst_seg = cur_shape_info['inst_seg']
        old_part_idx_to_new_part_idx = get_part_idx_to_num(inst_seg, mmax_nmask=mmax_nmask)
        new_inst_seg = np.zeros_like(inst_seg)
        for i in range(new_inst_seg.shape[0]):
            old_part_idx = int(inst_seg[i].item())
            new_part_idx = old_part_idx_to_new_part_idx[old_part_idx]
            new_inst_seg[i] = new_part_idx
        cur_new_shape_info = {"pos": cur_shape_info['pos'], 'inst_seg': new_inst_seg}
        new_shape_info[shp_idx] = cur_new_shape_info
    np.save(os.path.join(root_path, f"partnet_shapes_clip_partnum_{mmax_nmask}_{down_num_pts}.npy"), new_shape_info)


def split_by_shape_name(root_path):
    down_num_pts = 512
    mmax_nmask = 10
    shp_name_to_shp_idxes_file_name = "part_net_shp_cat_to_shp_anno_idx.npy"
    motion_part_info_file_name = f"partnet_shapes_clip_partnum_{mmax_nmask}_{down_num_pts}.npy"
    shp_name_to_shp_idxes = np.load(os.path.join(root_path, shp_name_to_shp_idxes_file_name), allow_pickle=True).item()
    motion_part_info = np.load(os.path.join(root_path, motion_part_info_file_name), allow_pickle=True).item()

    tot_root_path = os.path.join(root_path, f"partnet_shapes_clip_partnum_{mmax_nmask}_{down_num_pts}")
    if not os.path.exists(tot_root_path):
        os.mkdir(tot_root_path)

    for shp_name in shp_name_to_shp_idxes:
        # if shp_name in ["Cutting Instrument", "Earphone", "Laptop", "Refrigerator"]:
        #     continue
        act_name = shp_name
        if shp_name == "Hat/Cap":
            act_name = "Hat"
        print(f"Processing {shp_name}")
        cat_root_path = os.path.join(tot_root_path, act_name)
        if not os.path.exists(cat_root_path):
            os.mkdir(cat_root_path)
        cur_shp_idxes = shp_name_to_shp_idxes[shp_name]
        cur_shp_motion_part_info = {}
        for shp_idx in cur_shp_idxes:
            cur_motion_part_info = motion_part_info[shp_idx]
            cur_shp_motion_part_info[shp_idx] = cur_motion_part_info
        np.save(os.path.join(cat_root_path, "partnet_shapes_info.npy"), cur_shp_motion_part_info)

def split_by_shape_name_ori_set(root_path):
    down_num_pts = 512
    down_num_pts = 1024
    mmax_nmask = 10
    shp_name_to_shp_idxes_file_name = "part_net_shp_cat_to_shp_anno_idx.npy"
    motion_part_info_file_name = f"partnet_shapes_{down_num_pts}.npy"
    shp_name_to_shp_idxes = np.load(os.path.join(root_path, shp_name_to_shp_idxes_file_name), allow_pickle=True).item()
    motion_part_info = np.load(os.path.join(root_path, motion_part_info_file_name), allow_pickle=True).item()

    tot_root_path = os.path.join(root_path, f"partnet_shapes_{down_num_pts}")
    if not os.path.exists(tot_root_path):
        os.mkdir(tot_root_path)

    for shp_name in shp_name_to_shp_idxes:
        # if shp_name in ["Cutting Instrument", "Earphone", "Laptop", "Refrigerator"]:
        #     continue
        act_name = shp_name
        if shp_name == "Hat/Cap":
            act_name = "Hat"
        print(f"Processing {shp_name}")
        cat_root_path = os.path.join(tot_root_path, act_name)
        if not os.path.exists(cat_root_path):
            os.mkdir(cat_root_path)
        cur_shp_idxes = shp_name_to_shp_idxes[shp_name]
        cur_shp_motion_part_info = {}
        for shp_idx in cur_shp_idxes:
            cur_motion_part_info = motion_part_info[shp_idx]
            cur_shp_motion_part_info[shp_idx] = cur_motion_part_info
        np.save(os.path.join(cat_root_path, "partnet_shapes_info.npy"), cur_shp_motion_part_info)

def split_by_shape_name_ori_set_10000(root_path):
    down_num_pts = 512
    down_num_pts = 1024
    down_num_pts = 10000
    mmax_nmask = 10
    shp_name_to_shp_idxes_file_name = "part_net_shp_cat_to_shp_anno_idx.npy"
    motion_part_info_file_name = f"partnet_shapes_{down_num_pts}.npy"
    motion_part_info_file_name = f"tot_part_motion_meta_info.npy"
    shp_name_to_shp_idxes = np.load(os.path.join(root_path, shp_name_to_shp_idxes_file_name), allow_pickle=True).item()
    motion_part_info = np.load(os.path.join(root_path, motion_part_info_file_name), allow_pickle=True).item()

    tot_root_path = os.path.join(root_path, f"partnet_shapes_{down_num_pts}")
    if not os.path.exists(tot_root_path):
        os.mkdir(tot_root_path)

    for shp_name in shp_name_to_shp_idxes:
        # if shp_name in ["Cutting Instrument", "Earphone", "Laptop", "Refrigerator"]:
        #     continue
        act_name = shp_name
        if shp_name == "Hat/Cap":
            act_name = "Hat"
        print(f"Processing {shp_name}")
        cat_root_path = os.path.join(tot_root_path, act_name)
        if not os.path.exists(cat_root_path):
            os.mkdir(cat_root_path)
        cur_shp_idxes = shp_name_to_shp_idxes[shp_name]
        cur_shp_motion_part_info = {}
        for shp_idx in cur_shp_idxes:
            cur_motion_part_info = motion_part_info[shp_idx]
            cur_shp_motion_part_info[shp_idx] = cur_motion_part_info
        np.save(os.path.join(cat_root_path, "partnet_shapes_info.npy"), cur_shp_motion_part_info)

def split_into_train_val_test(root_path, not_cliped=True):
    down_num_pts = 512
    down_num_pts = 1024
    down_num_pts = 10000
    mmax_nmask = 10
    if not_cliped:
        shapes_info_folder_name = f"partnet_shapes_{down_num_pts}"
    else:
        shapes_info_folder_name = f"partnet_shapes_clip_partnum_{mmax_nmask}_{down_num_pts}"
    tot_shape_names = os.listdir(os.path.join(root_path, shapes_info_folder_name))
    for shp_name in tot_shape_names:
        pth = os.path.join(root_path, shapes_info_folder_name, shp_name, "partnet_shapes_info.npy")
        tot_part_meta_info = np.load(pth, allow_pickle=True).item()
        shp_idxes = list(tot_part_meta_info.keys())
        tot_num = len(shp_idxes)
        train_num, val_num = int(tot_num * 0.7), int(tot_num * 0.1)
        tst_num = tot_num - train_num - val_num
        print(shp_name, train_num, val_num, tst_num)

        permidx = np.random.permutation(tot_num)
        train_idxes = permidx[:train_num].tolist()
        val_idxes = permidx[train_num: train_num + val_num].tolist()
        tst_idxes = permidx[train_num + val_num: ].tolist()

        train_idxes = [shp_idxes[ii] for ii in train_idxes]
        val_idxes = [shp_idxes[ii] for ii in val_idxes]
        tst_idxes = [shp_idxes[ii] for ii in tst_idxes]

        train_part_meta_info = {shp_idx: tot_part_meta_info[shp_idx] for shp_idx in train_idxes}
        val_part_meta_info = {shp_idx: tot_part_meta_info[shp_idx] for shp_idx in val_idxes}
        tst_part_meta_info = {shp_idx: tot_part_meta_info[shp_idx] for shp_idx in tst_idxes}
        np.save(os.path.join(root_path, shapes_info_folder_name, shp_name,
                             "train_partnet_shapes_info.npy"), train_part_meta_info)
        np.save(os.path.join(root_path, shapes_info_folder_name, shp_name,
                             "val_partnet_shapes_info.npy"), val_part_meta_info)
        np.save(os.path.join(root_path, shapes_info_folder_name, shp_name,
                             "tst_partnet_shapes_info.npy"), tst_part_meta_info)

def split_data_via_dist(root):
    maxx_prms = 10
    n_clusters = 17
    file_names = ["train_data.txt", "test_data.txt", "val_data.txt"]
    fn_name_list = []
    fn_data_dist_list = []
    for fn in file_names:
        data_path = open(os.path.join(root, fn), "r")
        file_list = [item.strip() for item in data_path.readlines()]
        for data_file_name in file_list:
            data_file = os.path.join(root, data_file_name + '.h5')
            with h5py.File(data_file, 'r') as hf:
                primitives = np.array(hf.get("prim"))
            dist = np.zeros((maxx_prms, ), dtype=np.float)
            for pi in range(primitives.shape[0]):
                cur_prm = int(primitives[pi].item())
                dist[cur_prm] += 1
            dist = dist / np.sum(dist)
            fn_data_dist_list.append(dist)
            fn_name_list.append(data_file_name)

    # fn_name_list; fn_data_list_list
    data_dist_arr = np.array(fn_data_dist_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_dist_arr)
    labels = kmeans.labels_
    label_to_file_names = {}
    label_to_prm_dist = {}
    for i in range(labels.shape[0]):
        cur_label = int(labels[i].item())
        cur_file_name = fn_name_list[i]
        if cur_label not in label_to_file_names:
            label_to_file_names[cur_label] = [cur_file_name]
            label_to_prm_dist[cur_label] = [data_dist_arr[i]]
        else:
            label_to_file_names[cur_label].append(cur_file_name)
            label_to_prm_dist[cur_label].append(data_dist_arr[i])

    sim_nns = []
    label_to_dist_array = {}
    for cur_label in label_to_file_names:
        file_name_list = label_to_file_names[cur_label]
        nns = len(file_name_list)
        cur_label_data_dist_arr = np.array(label_to_prm_dist[cur_label])
        label_to_dist_array[cur_label] = cur_label_data_dist_arr
        dist_sims = np.matmul(cur_label_data_dist_arr, cur_label_data_dist_arr.T) / (np.sqrt(np.sum(cur_label_data_dist_arr ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(cur_label_data_dist_arr ** 2, axis=1, keepdims=False)[None, :]))
        dist_sims = (np.sum(dist_sims).item() - float(nns)) / (nns ** 2 - nns)
        print(cur_label, dist_sims)
        sim_nns.append(dist_sims)
        print(cur_label, len(file_name_list))
        with open(os.path.join(root, f"clustered_{n_clusters}_primitive_{cur_label}_data_names.txt"), "w") as wf:
            for fn in file_name_list:
                wf.write(fn + "\n")
            wf.close()
    print(sim_nns)
    print("sim_nns.mean =", np.mean(np.array(sim_nns)).item(), "sim_nns.var =", np.var(np.array(sim_nns)).item())

    ''' GET distribution similarity & difference average values '''
    label_to_dist_sim = {}
    label_to_dist_diff = {}
    for label_a in label_to_dist_array:
        dist_array_a = label_to_dist_array[label_a]
        cur_label_dist_diff = []
        for label_b in label_to_dist_array:
            dist_array_b = label_to_dist_array[label_b]
            if label_a == label_b:
                dist_sims = np.matmul(dist_array_a, dist_array_b.T) / (
                            np.sqrt(np.sum(dist_array_a ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(
                        dist_array_b ** 2, axis=1, keepdims=False)[None, :]))
                nns = dist_array_a.shape[0]
                dist_sims = (np.sum(dist_sims).item() - float(nns)) / float(nns ** 2 - nns)
                label_to_dist_sim[label_a] = dist_sims
            else:
                dist_sims = np.matmul(dist_array_a, dist_array_b.T) / (
                        np.sqrt(np.sum(dist_array_a ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(
                    dist_array_b ** 2, axis=1, keepdims=False)[None, :]))
                nns_a, nns_b = dist_array_a.shape[0], dist_array_b.shape[0]
                dist_sims = (np.sum(dist_sims).item()) / float(nns_a * nns_b)
                cur_label_dist_diff.append(dist_sims)
        label_to_dist_diff[label_a] = cur_label_dist_diff
    for cur_label in label_to_dist_diff:
        label_to_dist_diff[cur_label] = sum(label_to_dist_diff[cur_label]) / float(len(label_to_dist_diff[cur_label]))
    tot_sims = [label_to_dist_sim[cur_lab] for cur_lab in label_to_dist_sim]
    tot_difs = [label_to_dist_diff[cur_lab] for cur_lab in label_to_dist_diff]
    print(sum(tot_sims) / len(tot_sims), sum(tot_difs) / len(tot_difs))


def create_unit_data_from_hdf5(f, n_max_instances):
    '''
        f will be a h5py group-like object
    '''
    import re
    from datasets.data_util import create_primitive_from_dict, NAME_TO_ID_DICT

    P = f['gt_points'][()]  # Nx3
    normal_gt = f['gt_normals'][()]  # Nx3
    I_gt = f['gt_labels'][()]  # N


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

    T_gt = [NAME_TO_ID_DICT[primitive['type']] for primitive in instances]
    # SET the primitive type of the backgroud  part to `-1`
    T_gt = T_gt + [-1] + [0 for _ in range(n_max_instances - n_instances)]
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
    per_point_prim = T_gt[I_gt]

    return per_point_prim

def get_dists_clusters(root_path):
    n_clusters = 7
    n_max_primitives = 4
    for cur_label in range(n_clusters):
        fn = f"clustered_{n_clusters}_primitive_{cur_label}_data_names.txt"
        fl_list = []
        with open(os.path.join(root_path, fn), "r") as rf:
            for line in rf:
                cur_fn_name = line.strip()
                fl_list.append(cur_fn_name)

        print(cur_label, len(fl_list))
        continue
        cur_label_prim_arr = []
        # print(fl_list)
        for fn in fl_list:
            # print(root_path, fn)
            f = h5py.File(os.path.join(root_path, fn), "r")
            per_point_prim = create_unit_data_from_hdf5(f, 24)
            prim_to_nn = np.zeros((n_max_primitives,), dtype=np.float, )
            for i in range(per_point_prim.shape[0]):
                cur_prim = int(per_point_prim[i].item())
                if cur_prim != -1:
                    prim_to_nn[cur_prim] += 1
            prim_to_nn = prim_to_nn / np.sum(prim_to_nn)
            cur_label_prim_arr.append(prim_to_nn)
        cur_label_prim_arr = np.array(cur_label_prim_arr)
        cur_label_prim_arr = np.mean(cur_label_prim_arr, axis=0)
        print(cur_label, cur_label_prim_arr)



def split_data_via_dist_ANSI(root_path):
    all_h5_files = []
    train_test_file_names = ["train_models.csv", "test_models.csv"]
    n_clusters = 7 # 4

    for tr_file_name in train_test_file_names:
        csv_raw = pandas.read_csv(os.path.join(root_path, tr_file_name), delimiter=',', header=None)
        hdf5_file_list = list(csv_raw[0])
        all_h5_files += hdf5_file_list

    n_max_primitives = 4 # 5
    prim_to_nn_dist = []

    for file_name in all_h5_files:
        # print(root_path, file_name)
        # if not file_name.endswith(".h5"):
        #     continue
        f = h5py.File(os.path.join(root_path, file_name), 'r')
        per_point_prim = create_unit_data_from_hdf5(f, 24)
        prim_to_nn = np.zeros((n_max_primitives,), dtype=np.float, )
        for i in range(per_point_prim.shape[0]):
            cur_prim = int(per_point_prim[i].item())
            if cur_prim != -1:
                prim_to_nn[cur_prim] += 1
        prim_to_nn = prim_to_nn / np.sum(prim_to_nn)
        prim_to_nn_dist.append(prim_to_nn)

    data_dist_arr = np.array(prim_to_nn_dist)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_dist_arr)
    labels = kmeans.labels_
    label_to_file_names = {}
    label_to_prm_dist = {}
    for i in range(labels.shape[0]):
        cur_label = int(labels[i].item())
        cur_file_name = all_h5_files[i]
        if cur_label not in label_to_file_names:
            label_to_file_names[cur_label] = [cur_file_name]
            label_to_prm_dist[cur_label] = [data_dist_arr[i]]
        else:
            label_to_file_names[cur_label].append(cur_file_name)
            label_to_prm_dist[cur_label].append(data_dist_arr[i])

    sim_nns = []
    label_to_dist_array = {}
    for cur_label in label_to_file_names:
        file_name_list = label_to_file_names[cur_label]
        nns = len(file_name_list)
        cur_label_data_dist_arr = np.array(label_to_prm_dist[cur_label])
        label_to_dist_array[cur_label] = cur_label_data_dist_arr
        dist_sims = np.matmul(cur_label_data_dist_arr, cur_label_data_dist_arr.T) / (
                    np.sqrt(np.sum(cur_label_data_dist_arr ** 2, axis=1, keepdims=True)) * np.sqrt(
                np.sum(cur_label_data_dist_arr ** 2, axis=1, keepdims=False)[None, :]))
        dist_sims = (np.sum(dist_sims).item() - float(nns)) / (nns ** 2 - nns)
        print(cur_label, dist_sims)
        sim_nns.append(dist_sims)
        print(cur_label, len(file_name_list))
        with open(os.path.join(root_path, f"clustered_{n_clusters}_primitive_{cur_label}_data_names.txt"), "w") as wf:
            for fn in file_name_list:
                wf.write(fn + "\n")
            wf.close()
    print(sim_nns)
    print("sim_nns.mean =", np.mean(np.array(sim_nns)).item(), "sim_nns.var =", np.var(np.array(sim_nns)).item())

    ''' GET distribution similarity & difference average values '''
    label_to_dist_sim = {}
    label_to_dist_diff = {}
    for label_a in label_to_dist_array:
        dist_array_a = label_to_dist_array[label_a]
        cur_label_dist_diff = []
        for label_b in label_to_dist_array:
            dist_array_b = label_to_dist_array[label_b]
            if label_a == label_b:
                dist_sims = np.matmul(dist_array_a, dist_array_b.T) / (
                        np.sqrt(np.sum(dist_array_a ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(
                    dist_array_b ** 2, axis=1, keepdims=False)[None, :]))
                nns = dist_array_a.shape[0]
                dist_sims = (np.sum(dist_sims).item() - float(nns)) / float(nns ** 2 - nns)
                label_to_dist_sim[label_a] = dist_sims
            else:
                dist_sims = np.matmul(dist_array_a, dist_array_b.T) / (
                        np.sqrt(np.sum(dist_array_a ** 2, axis=1, keepdims=True)) * np.sqrt(np.sum(
                    dist_array_b ** 2, axis=1, keepdims=False)[None, :]))
                nns_a, nns_b = dist_array_a.shape[0], dist_array_b.shape[0]
                dist_sims = (np.sum(dist_sims).item()) / float(nns_a * nns_b)
                cur_label_dist_diff.append(dist_sims)
        label_to_dist_diff[label_a] = cur_label_dist_diff
    for cur_label in label_to_dist_diff:
        label_to_dist_diff[cur_label] = sum(label_to_dist_diff[cur_label]) / float(len(label_to_dist_diff[cur_label]))
    tot_sims = [label_to_dist_sim[cur_lab] for cur_lab in label_to_dist_sim]
    tot_difs = [label_to_dist_diff[cur_lab] for cur_lab in label_to_dist_diff]
    print(sum(tot_sims) / len(tot_sims), sum(tot_difs) / len(tot_difs))


def test_dataset_param(root):
    file_list = os.listdir(root)
    for fn in file_list:
        if fn.endswith(".h5"):
            data_file = os.path.join(root, fn)

            with h5py.File(data_file, 'r') as hf:
                points = np.array(hf.get("points"))
                labels = np.array(hf.get("labels"))  # each point's label
                normals = np.array(hf.get("normals"))
                primitives = np.array(hf.get("prim"))
                primitive_param = np.array(hf.get("T_param"))

            print(primitive_param.shape, points.shape, labels.shape, primitives.shape, normals.shape)
            # break

            # set small number primitive as background
            counter = Counter(labels)
            mapper = np.ones([labels.max() + 1]) * -1
            # # k --- label idx
            keys = [k for k, v in counter.items() if v > 100]
            keys = sorted(keys, key=lambda i: counter[i], reverse=True)
            #
            maximum_key = keys[0]

            for maximum_key in keys:
                kk = 0

                cur_inst_prams = []
                for j in range(labels.shape[0]):
                    cur_inst_label = int(labels[j].item())
                    if cur_inst_label == maximum_key:
                        # print(primitive_param[j])
                        kk += 1
                        cur_inst_prams.append(primitive_param[j])
                        # if kk >= 10:
                        #     break

                ltwo_dist = []
                for i in range(len(cur_inst_prams)):
                    for j in range(i + 1, len(cur_inst_prams)):
                        cur_dist = np.sum((cur_inst_prams[i] - cur_inst_prams[j]) ** 2)
                        ltwo_dist.append(cur_dist.item())
                print(max(ltwo_dist), min(ltwo_dist), sum(ltwo_dist) / len(ltwo_dist))
            break

            # # for i in range()
            #
            # # keys --- as indexes
            # # len(keys) ---
            # if len(keys):
            #     mapper[keys] = np.arange(
            #         len(keys))  # reindex label idxes --- valid labels are re-indexed and invalid labels are set to -1
            # label = mapper[labels]  # label vector stores valid labels of each point in the shape
            # # index the array
            # # I_gt --- I_gt --- labels --- labeldic
            # ret_dict['I_gt'] = label.astype(int)
            # # get clean primitives
            # clean_primitives = np.ones_like(primitives) * -1
            # valid_mask = label != -1
            # # select clean primitives
            # clean_primitives[valid_mask] = primitives[valid_mask]
            # ret_dict['T_gt'] = clean_primitives.astype(int)
            #
            # ret_dict['index'] = self.data_list[index]  # index --- only the unique index of each object
            #
            # small_idx = label == -1
            # full_labels = label
            # # all idxes ---- to ---- len(keys) - 1
            # # labels[small_idx]
            # # full_labels[small_idx] = labels[small_idx] + len(keys)
            # full_labels[small_idx] = len(keys)
            # # ret_dict['I_gt_clean'] = self.reindex_labels(full_labels.astype(int))
            # ret_dict['I_gt_clean'] = full_labels.astype(int)
            # # return points, normals
            # # valid masks --- number of valid masks
            # ret_dict['curnmask'] = np.array([len(keys)], dtype=np.long)
            # return ret_dict


import random
def get_simple_test_examples(root):
    ansi_n_clusters = 7
    tst_data_nn_each_prim = 10
    prim_types = [i for i in range(0, ansi_n_clusters)]
    n_clusters = ansi_n_clusters
    prim_types = [int(item) for item in prim_types if item != ',']
    data_list = []
    for prm_idx in prim_types:
        prm_file_name = f"clustered_{n_clusters}_primitive_{prm_idx}_data_names.txt"
        prm_data_file_path = open(os.path.join(root, prm_file_name), "r")
        prm_data_list = [item.strip() for item in prm_data_file_path.readlines()]
        random.shuffle(prm_data_list)
        selected_test_data = prm_data_list[:tst_data_nn_each_prim]
        with open(os.path.join(root, f"clustered_{n_clusters}_primitive_{prm_idx}_data_names_inference.txt"), "w") as wf:
            for tst_dt in selected_test_data:
                wf.write(tst_dt + "\n")
            wf.close()

def test_motion_dataset(root):
    split = "val"
    fl_path = os.path.join(root, f"all_type_tot_{split}_tot_part_motion_meta_info.npy")
    data = np.load(fl_path, allow_pickle=True).item()
    ii = 0
    for k in data:
        print(ii, k)
        ii += 1
        if ii >= 10:
            break




if __name__ == '__main__':
    # root_path = "/data-input"
    root_path = "./data/traceparts_data"

    # down_sample_partnet(root_path)
    # clip_part_num(root_path)
    # split_by_shape_name(root_path)
    # split_into_train_val_test(root_path)
    # split_by_shape_name_ori_set(root_path)
    # split_into_train_val_test(root_path)
    # down_sample_partnet(root_path)
    # split_by_shape_name_ori_set(root_path)
    # split_into_train_val_test(root_path)
    # split_by_shape_name_ori_set_10000(root_path)
    # split_data_via_dist(root_path)
    # split_data_via_dist_ANSI(root_path)
    # test_dataset_param(root_path)
    # get_dists_clusters(root_path)
    # get_simple_test_examples(root_path)
    test_motion_dataset(root_path)
