import scipy.io as sio
import os
import numpy as np


try:
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at((0., 0.0, 1.5), (0., 0., 1.))
    ps.set_screenshot_extension(".png")
except:
    pass


color = [
    (136/255.0,224/255.0,239/255.0),
    (180/255.0,254/255.0,152/255.0),
    (184/255.0,59/255.0,94/255.0),
    (106/255.0,44/255.0,112/255.0),
    (39/255.0,53/255.0,135/255.0),
(0,173/255.0,181/255.0), (170/255.0,150/255.0,218/255.0), (82/255.0,18/255.0,98/255.0), (234/255.0,84/255.0,85/255.0), (234/255.0,255/255.0,208/255.0),(162/255.0,210/255.0,255/255.0),
    (187/255.0,225/255.0,250/255.0), (240/255.0,138/255.0,93/255.0), (184/255.0,59/255.0,94/255.0),(106/255.0,44/255.0,112/255.0),(39/255.0,53/255.0,135/255.0),
]


def mat_values(fn, save_fn="sf2f_test_est-1.mat"):
    data = sio.loadmat(fn)
    print(data.keys())

    keys = ['pc1', 'seg1', 'flow_est']

    for k in keys:
        print(k, data[k].shape)
    # print(data['pc1'].shape)
    pc1, seg1, flow_est = data['pc1'][:].astype(np.float32), data['seg1'][:].astype(np.int32), data['flow_est'][:].astype(np.float32)

    i_bz = 0
    cur_pc1, cur_seg1, cur_flow_est = pc1[i_bz], seg1[i_bz], flow_est[i_bz]
    seg_idx_to_pts_idx = {}
    for i_pts in range(cur_seg1.shape[0]):
        cur_seg_idx = int(cur_seg1[i_pts].item())
        if cur_seg_idx not in seg_idx_to_pts_idx:
            seg_idx_to_pts_idx[cur_seg_idx] = [i_pts]
        else:
            seg_idx_to_pts_idx[cur_seg_idx].append(i_pts)
    for cur_seg_idx in seg_idx_to_pts_idx:
        seg_idx_to_pts_idx[cur_seg_idx] = np.array(seg_idx_to_pts_idx[cur_seg_idx], dtype=np.long)
    for cur_seg_idx in seg_idx_to_pts_idx:
        cur_seg_pts_idxes = seg_idx_to_pts_idx[cur_seg_idx]
        cur_seg_pts = cur_pc1[cur_seg_pts_idxes]
        ps.register_point_cloud(f"pc1_{cur_seg_idx}", cur_seg_pts, radius=0.012, color=color[cur_seg_idx])
    ps.show()
    ps.remove_all_structures()

    for cur_seg_idx in seg_idx_to_pts_idx:
        cur_seg_pts_idxes = seg_idx_to_pts_idx[cur_seg_idx]
        cur_seg_pts = (cur_pc1 + cur_flow_est)[cur_seg_pts_idxes]
        ps.register_point_cloud(f"pc1_{cur_seg_idx}", cur_seg_pts, radius=0.012, color=color[cur_seg_idx])
    ps.show()
    ps.remove_all_structures()

    # new_pc1 = []
    # new_seg1 = []
    # new_flow_est = []
    # for i_bz in range(pc1.shape[0]):
    #     rnd_idx = np.random.permutation(pc1[i_bz].shape[0])
    #     print(i_bz, rnd_idx.shape, np.min(rnd_idx), np.max(rnd_idx))
    #     cur_n_pc1, cur_n_seg1, cur_n_flow_est = pc1[i_bz][rnd_idx], seg1[i_bz][rnd_idx], flow_est[i_bz][rnd_idx]
    #     new_pc1.append(cur_n_pc1)
    #     new_seg1.append(cur_n_seg1)
    #     new_flow_est.append(cur_n_flow_est)
    # new_pc1 = np.array(new_pc1, dtype=np.float32)
    # new_seg1 = np.array(new_seg1, dtype=np.int32)
    # new_flow_est = np.array(new_flow_est, dtype=np.float32)


    # new_data = {
    #     'pc1': new_pc1,
    #     # 'pc2': data['pc2'],
    #     'seg1': new_seg1,
    #     'flow_est': new_flow_est
    # }
    # sio.savemat(save_fn, new_data)


if __name__=='__main__':
    fn = os.path.join("./data", "part-segmentation", "sf2f_test_est.mat")

    save_fn = os.path.join("./data", "part-segmentation", "sf2f_test_est-1.mat")
    fn = save_fn
    mat_values(fn=fn, save_fn=save_fn)

# baseline_loss_dict = [{'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 4}, 'rgt_chd': {'uop': 1, 'oper': 2}}]
# motion-seg
# ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 4}, 'rgt_chd': {'uop': 1, 'oper': 2}}, 0.00029583821151221663), ({'gop': 1, 'uop': 2, 'chd': {'uop': 4, 'oper': 0}}, 0.0002958248792966858), ({'gop': 1, 'uop': 2, 'chd': {'uop': 0, 'oper': 1}}, 0.0002958248792966858), ({'gop': 1, 'uop': 2, 'chd': {'uop': 5, 'oper': 0}}, 0.0002958248792966858), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 2, 'oper': 5}, 'rgt_chd': {'uop': 1, 'oper': 4}}, 0.00029280328517061646), ({'gop': 2, 'uop': 4, 'chd': {'uop': 1, 'oper': 0}}, 0.00020287625131291937), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 0, 'oper': 1}, 'rgt_chd': {'uop': 1, 'oper': 2}}, 0.0001409045146977492), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 5, 'oper': 4}, 'rgt_chd': {'uop': 1, 'oper': 4}}, 0.00010429186715753342), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 1, 'oper': 6}, 'rgt_chd': {'uop': 0, 'oper': 3}}, 8.91584890401881e-05), ({'gop': 1, 'uop': 4, 'chd': {'uop': 1, 'oper': 2}}, 5.127113495671487e-05), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 5, 'oper': 6}, 'rgt_chd': {'uop': 1, 'oper': 5}}, 4.942853312710812e-05), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 5, 'oper': 1}, 'rgt_chd': {'uop': 1, 'oper': 5}}, 4.942853312710812e-05), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 4, 'oper': 0}, 'rgt_chd': {'uop': 1, 'oper': 3}}, 4.8025049624744454e-05), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 4, 'oper': 4}, 'rgt_chd': {'uop': 3, 'oper': 1}}, 4.1582260633600025e-05), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 2, 'oper': 1}, 'rgt_chd': {'uop': 3, 'oper': 2}}, 3.50592301279459e-05), ({'gop': 3, 'uop': 1, 'bop': 2, 'lft_chd': {'uop': 5, 'oper': 4}, 'rgt_chd': {'uop': 3, 'oper': 1}}, 3.345221028673583e-05), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 0, 'oper': 4}, 'rgt_chd': {'uop': 2, 'oper': 5}}, 2.8187095207185766e-05), ({'gop': 3, 'uop': 1, 'bop': 3, 'lft_chd': {'uop': 3, 'oper': 6}, 'rgt_chd': {'uop': 0, 'oper': 3}}, 1.6323013631686e-05), ({'gop': 0, 'uop': 1, 'bop': 4, 'lft_chd': {'uop': 4, 'oper': 5}, 'rgt_chd': {'uop': 3, 'oper': 4}}, 1.3566756490296251e-05)

