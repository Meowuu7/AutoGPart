import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_util import construct_conv_modules, CorrFlowPredNet, construct_conv1d_modules
import numpy as np
from .utils import batched_index_select, farthest_point_sampling
try:
    import open3d as o3d
except:
    pass
import os
# import horovod.torch as hvd
import logging
from .constants import *
try:
    import horovod.torch as hvd
except:
    pass

try:
    from filelock import FileLock
except:
    pass

# todo: make dir for loss model save path
class ComputingGraphLossModel(nn.Module):
    def __init__(self, pos_dim, pca_feat_dim=64, in_feat_dim=64, pp_sim_k=16, r=0.03, lr_scaler=1.0, init_lr=0.001, weight_decay=1e-5, loss_model_save_path="", in_rep_dim=128, nn_inter_loss=1, args=None):
        super(ComputingGraphLossModel, self).__init__()

        self.feat_k = pos_dim
        self.pos_dim = pos_dim
        self.feat_by_feat_k = self.feat_k * self.feat_k
        self.pca_feat_k = pca_feat_dim

        self.level_one_feat_candi = 5

        self.level_two_feat_candi = (self.level_one_feat_candi ** 2) # * 2

        self.pp_sim_k = pp_sim_k # size of local neighbourhood
        self.r = r # radius

        self.args = args
        # self.rep_feat_dim = 32
        self.rep_feat_dim = in_rep_dim
        # ALL geometric features are 3-dim features
        self.in_feat_dim = 3
        # REMEMBER `loss_model_save_path`
        self.loss_model_save_path = loss_model_save_path
        # WHETHER with spectral operation
        self.no_spectral = args.no_spectral
        # NUMBER of added intermediate losses
        self.nn_inter_loss = nn_inter_loss

        self.hvd = True  if "_h_" in args.task  else False
        self.dist = True if "_dist_" in args.task else False
        self.local_rank = args.local_rank
        self.test_performance = args.test_performance

        ''' GET all prediction heads '''
        # SET operator number from args.
        self.nn_grp_opers = args.nn_grp_opers # 4  # bz x N x k x f... -> bz x N x f...: sum, mean, max, svd
        self.nn_binary_opers = args.nn_binary_opers # 6  # add, minus, element-wise multiply, cross product, cartesian product, matrix-vector product
        self.nn_unary_opers = args.nn_unary_opers # 7  # identity, square, 2, -, -2, inv, orth
        self.nn_in_feats = args.nn_in_feats # 2

        print("Start visiting tree === ")

        ''' GET all prediction heads '''

        self.nn_pred_feat_dims = [1, 3, 9]
        ''' GET pred_feat_dim to first appearence index '''
        self.pred_feat_dim_to_head_idx = {}
        for pred_feat_dim in self.nn_pred_feat_dims:
            if pred_feat_dim not in self.pred_feat_dim_to_head_idx:
                cur_idx = len(self.pred_feat_dim_to_head_idx)
                self.pred_feat_dim_to_head_idx[pred_feat_dim] = cur_idx
        ''' GET pred_feat_dim to first appearence index '''

        ''' GET in-memory heads '''
        self.pred_feat_dim_to_head_list = []
        for i in range(self.nn_inter_loss):
            ''' SET all prediction heads; SET fake prediction head '''
            real_head_list = []

            for pred_dim in self.pred_feat_dim_to_head_idx:
                tmp = construct_conv1d_modules(
                    [self.rep_feat_dim // 2, pred_dim], n_in=self.rep_feat_dim, last_act=False, bn=False
                )
                real_head_list.append(tmp)
            ''' SET all prediction heads; SET fake prediction head '''

            real_head_list = nn.ModuleList(real_head_list)
            self.pred_feat_dim_to_head_list.append(real_head_list)
            # self.pred_feat_dim_to_head_list.append(pred_feat_dim_to_head)

        self.pred_feat_dim_to_head_list = nn.ModuleList(self.pred_feat_dim_to_head_list)
        ''' GET in-memory heads '''

        ''' SET useful dictionaries '''
        self.selected_oper_dict_to_module_index = {}
        self.module_index_to_selected_oper_dict = {}
        self.module_index_to_head_idx_dict = {}
        ''' SET useful dictionaries '''


    def mult_dim_list(self, dim_list):
        curr_dim = 1
        for jj in dim_list:
            curr_dim *= jj
        return curr_dim

    def get_bop_dim_list_by_uops_list(self, lft_chd_dim, rgt_chd_dim, j_binary):
        ans_dim_list = None
        if j_binary in [0, 1, 2]:
            if len(lft_chd_dim) == len(rgt_chd_dim):
                equ = True
                for kk, (lftt_dim, rgtt_dim) in enumerate(zip(lft_chd_dim, rgt_chd_dim)):
                    if lftt_dim != rgtt_dim:
                        equ = False
                        break
                if equ:
                    ans_dim_list = lft_chd_dim
        elif j_binary in [3]:
            if len(lft_chd_dim) == 1 and len(rgt_chd_dim) == 1 and lft_chd_dim[0] == rgt_chd_dim[0]:
                ans_dim_list = lft_chd_dim
        elif j_binary in [4]:
            if len(lft_chd_dim) == 1 and len(rgt_chd_dim) == 1:
                ans_dim_list = [lft_chd_dim[0], rgt_chd_dim[0]]
        elif j_binary in [5]:
            if len(lft_chd_dim) == 2 and len(rgt_chd_dim) == 2 and lft_chd_dim[1] == rgt_chd_dim[0]:
                ans_dim_list = [lft_chd_dim[0], rgt_chd_dim[1]]
        return ans_dim_list

    # visit operator tree --- this only aims to get possible operators and their corresponding feature dimensions
    # todo: we should not include `no-grouping` in grouping operator, thus it can better describe available operators
    def visit_operator_tree(self, depth=0):
        rt_list = []
        rt_dim_list = []

        if depth == 2:
            # print("Got an leaf node === ")
            # the leaf node only has an input feature selection node and a unary operator for feature transformation
            # thus we should remember
            for i_unary in range(self.nn_unary_opers):
                for j in range(self.nn_in_feats):
                    # leaf node has no grouping operation
                    rt_list.append({"uop": i_unary, "oper": j})
                    rt_dim_list.append([self.in_feat_dim])
        else:
            for i_unary in range(self.nn_unary_opers):
                for i_grp in range(self.nn_grp_opers):
                    # self.dict = {"uop": i_unary, "gop": i_grp}
                    # sample unary operators
                    chd_dict_list, chd_dim_list = self.visit_operator_tree(2)
                    print(f"length of chd_dict_list for uanry children = {len(chd_dict_list)}")
                    for chd_dict, chd_dim in zip(chd_dict_list, chd_dim_list):
                        cur_des_dict = {"uop": i_unary, "gop": i_grp, "chd": chd_dict}
                        rt_list.append(cur_des_dict)
                        if i_unary == self.nn_unary_opers - 1 and len(chd_dim) == 2:
                            rt_dim_list.append(list(reversed(chd_dim)))
                        else:
                            rt_dim_list.append(chd_dim)
                    if depth == 1 or depth == 0:
                        for j_binary in range(self.nn_binary_opers):
                            for lft_chd_dict, lft_chd_dim in zip(chd_dict_list, chd_dim_list):
                                for rgt_chd_dict, rgt_chd_dim in zip(chd_dict_list, chd_dim_list):
                                    cur_dim_list = self.get_bop_dim_list_by_uops_list(lft_chd_dim, rgt_chd_dim, j_binary)
                                    if cur_dim_list is not None:
                                        cur_des_dict = {"gop": i_grp, "uop": i_unary, "bop": j_binary + 1, "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
                                        rt_list.append(cur_des_dict)
                                        rt_dim_list.append(cur_dim_list)
                    if depth == 0:
                        grp_chd_dict_list, grp_chd_dim_list = self.visit_operator_tree(1)
                        print(f"length of grp_chd_dict_list for binary children = {len(grp_chd_dict_list)}")
                        for j_binary in range(self.nn_binary_opers):
                            for lft_chd_dict, lft_chd_dim in zip(grp_chd_dict_list, grp_chd_dim_list):
                                for rgt_chd_dict, rgt_chd_dim in zip(chd_dict_list, chd_dim_list):
                                    cur_dim_list = self.get_bop_dim_list_by_uops_list(lft_chd_dim, rgt_chd_dim, j_binary)
                                    if cur_dim_list is not None:
                                        cur_des_dict = {"gop": i_grp, "uop": i_unary, "bop": self.nn_binary_opers + j_binary + 1, "lft_chd": lft_chd_dict,
                                                        "rgt_chd": rgt_chd_dict}
                                        rt_list.append(cur_des_dict)
                                        rt_dim_list.append(cur_dim_list)
                            for lft_chd_dict, lft_chd_dim in zip(chd_dict_list, chd_dim_list):
                                for rgt_chd_dict, rgt_chd_dim in zip(grp_chd_dict_list, grp_chd_dim_list):
                                    cur_dim_list = self.get_bop_dim_list_by_uops_list(lft_chd_dim, rgt_chd_dim, j_binary)
                                    if cur_dim_list is not None:
                                        cur_des_dict = {"gop": i_grp, "uop": i_unary, "bop": self.nn_binary_opers * 2 + j_binary + 1, "lft_chd": lft_chd_dict,
                                                        "rgt_chd": rgt_chd_dict}
                                        rt_list.append(cur_des_dict)
                                        rt_dim_list.append(cur_dim_list)
        return rt_list, rt_dim_list

    def visit_operator_tree_count_only(self, depth=0):
        # rt_list = []
        # rt_dim_list = []
        tot_nn = 0
        if depth == 2:
            # print("Got an leaf node === ")
            for i_grp in range(self.nn_grp_opers + 1):
                for i_unary in range(self.nn_unary_opers):
                    for j in range(self.nn_in_feats):
                        tot_nn += 1
        else:
            n_sampling_grp_oper = self.nn_grp_opers + 1 if depth > 0 else self.nn_grp_opers
            for i_grp in range(n_sampling_grp_oper):
                for j_unary in range(self.nn_unary_opers):
                    chd_nn = self.visit_operator_tree_count_only(depth + 1)
                    tot_nn += chd_nn
                for j_binary in range(self.nn_binary_opers):
                    chd_nn = self.visit_operator_tree_count_only(depth + 1)
                    tot_nn += chd_nn ** 2

        return tot_nn

    ''' GET resulting feature dimension from `oper_dict` '''
    @staticmethod
    def get_pred_feat_dim_from_oper_dict(oper_dict, in_feat_dim=3, nn_binary_opers=3):
        rt_dim = []
        if "oper" in oper_dict:
            rt_dim = [in_feat_dim]
        elif "chd" in oper_dict:
            # GET child's feature dimension
            # chd_dim = self.get_pred_feat_dim_from_oper_dict(oper_dict["chd"])
            chd_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict["chd"], in_feat_dim=in_feat_dim, nn_binary_opers=nn_binary_opers)
            # GET child's feature dimension after the unary transformation
            uop = oper_dict["uop"]
            # ONLY INVERSE will change the feature dimension of the resulting feature matrix
            if uop == INVERSE and len(chd_dim) == 2:
                rt_dim = list(reversed(chd_dim))
            else:
                rt_dim = chd_dim
            # GET feature dimension after grouping transformation
            gop = oper_dict["gop"]

            if gop == SVD and (len(rt_dim) > 1):
                rt_dim = None
            elif gop == SVD:
                rt_dim = [rt_dim[-1], rt_dim[-1]]
            # if gop == SVD and
            if gop >= GROUP_INVALID or uop >= UNARY_INVALID:
                rt_dim = None

        else:
            # lft_chd_dim, rgt_chd_dim = self.get_pred_feat_dim_from_oper_dict(oper_dict["lft_chd"]), self.get_pred_feat_dim_from_oper_dict(oper_dict["rgt_chd"])
            lft_chd_dim, rgt_chd_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict["lft_chd"], in_feat_dim=in_feat_dim, nn_binary_opers=nn_binary_opers), ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict["rgt_chd"], in_feat_dim=in_feat_dim, nn_binary_opers=nn_binary_opers)
            if lft_chd_dim is not None and rgt_chd_dim is not None:
                # bop = oper_dict["bop"]
                bop = (oper_dict["bop"] - 1) % nn_binary_opers
                if bop in [ADD, MINUS, MULTIPLY]:
                    if len(lft_chd_dim) == len(rgt_chd_dim):
                        equ = True
                        for kk, (lftt_dim, rgtt_dim) in enumerate(zip(lft_chd_dim, rgt_chd_dim)):
                            if lftt_dim != rgtt_dim:
                                equ = False
                                break
                        if equ:
                            rt_dim = lft_chd_dim
                        else:
                            rt_dim = None
                    else:
                        rt_dim = None
                elif bop in [CROSS_PRODUCT]:
                    if len(lft_chd_dim) == 1 and len(rgt_chd_dim) == 1 and lft_chd_dim[0] == rgt_chd_dim[0]:
                        rt_dim = lft_chd_dim
                    else:
                        rt_dim = None
                elif bop in [CARTESIAN_PRODUCT]:
                    if len(lft_chd_dim) == 1 and len(rgt_chd_dim) == 1:
                        rt_dim = [lft_chd_dim[0], rgt_chd_dim[0]]
                    else:
                        rt_dim = None
                elif bop in [MATRIX_VECTOR_PRODUCT]:
                    if len(lft_chd_dim) == 2 and len(rgt_chd_dim) == 2 and lft_chd_dim[1] == rgt_chd_dim[0]:
                        rt_dim = [lft_chd_dim[0], rgt_chd_dim[1]]
                    else:
                        rt_dim = None
                else:
                    rt_dim = None
            else:
                rt_dim = None

            if rt_dim is not None:
                uop, gop = oper_dict["uop"], oper_dict["gop"]
                if uop == INVERSE and len(rt_dim) == 2:
                    rt_dim = list(reversed(rt_dim))
                if gop == SVD and (len(rt_dim) > 1):
                    rt_dim = None
                elif gop == SVD:
                    rt_dim = [rt_dim[-1], rt_dim[-1]]
        return rt_dim

    # save heads
    ''' SAVE heads by selected operator dictionaries '''
    def save_head_optimizer_by_operation_dicts(self, oper_dicts, model_idx=None):
        if not self.hvd or (self.hvd and hvd.rank() == 0):
            for i in range(len(self.pred_feat_dim_to_head_list)):
                oper_dict = self.module_index_to_selected_oper_dict[i]
                if oper_dict is None:
                    continue
                head_descriptor = str(oper_dict)
                if model_idx is not None:
                    head_optim_save_file_name = f"{head_descriptor}_model_optim_dict_model_idx_{model_idx}.tar"
                else:
                    head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"

                # cur_head_pred_dim = self.get_pred_feat_dim_from_oper_dict(oper_dict)
                cur_head_pred_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict, in_feat_dim=self.in_feat_dim, nn_binary_opers=self.nn_binary_opers)
                cur_head_pred_dim = self.mult_dim_list(cur_head_pred_dim)
                cur_head_idx = self.pred_feat_dim_to_head_idx[cur_head_pred_dim]
                save_dict = {
                    "model": self.pred_feat_dim_to_head_list[i][cur_head_idx].state_dict()
                }
                torch.save(save_dict, os.path.join(self.loss_model_save_path, head_optim_save_file_name))

    # SAVE init head optimizer by operation dictionaries
    def save_init_head_optimizer_by_operation_dicts(self, oper_dicts):

        for i, oper_dict in enumerate(oper_dicts):
            pred_feat_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict, in_feat_dim=self.in_feat_dim, nn_binary_opers=self.nn_binary_opers)

            if pred_feat_dim is None:
                continue

            pred_feat_dim = self.mult_dim_list(pred_feat_dim)

            head_descriptor = str(oper_dict)
            head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"

            loss_model_pth_path = os.path.join(self.loss_model_save_path, head_optim_save_file_name)

            with FileLock(os.path.expanduser("~/.horovod_lock")):
                if not os.path.exists(loss_model_pth_path):
                    tmp_model = construct_conv1d_modules(
                        [self.rep_feat_dim // 2, pred_feat_dim], n_in=self.rep_feat_dim, last_act=False, bn=False
                    ).cuda()

                    head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"
                    head_states = {
                        "model": tmp_model.state_dict(),
                    }
                    torch.save(head_states, os.path.join(loss_model_pth_path))

    ''' LOAD head optimizer by multiple operation dictionaries '''
    def load_head_optimizer_by_operation_dicts(self, oper_dicts, init_lr=0.001, weight_decay=1e-4, model_idx=None):
        self.selected_oper_dict_to_module_index = {}
        self.module_index_to_selected_oper_dict = {}
        self.module_index_to_head_idx_dict = {}

        for i, oper_dict in enumerate(oper_dicts):
            print(f"Loading {i}-th oper dict: {oper_dict}")
            # part_oper = oper_dict["part_oper"]
            # if part_oper == self.nn_part_level_operations:
            #     continue
            # pred_feat_dim = self.get_pred_feat_dim_from_oper_dict(oper_dict)
            pred_feat_dim = ComputingGraphLossModel.get_pred_feat_dim_from_oper_dict(oper_dict, in_feat_dim=self.in_feat_dim, nn_binary_opers=self.nn_binary_opers)

            if pred_feat_dim is None:
                # Model index to the corresponding operation dictionary
                self.module_index_to_selected_oper_dict[i] = None
                continue

            # print(oper_dict)
            # print(i, pred_feat_dim)
            pred_feat_dim = self.mult_dim_list(pred_feat_dim)
            self.module_index_to_selected_oper_dict[i] = oper_dict

            cur_head_idx = self.pred_feat_dim_to_head_idx[pred_feat_dim]

            self.module_index_to_head_idx_dict[i] = cur_head_idx

            if self.test_performance:
                continue

            head_descriptor = str(oper_dict)
            if model_idx is not None:
                head_optim_save_file_name = f"{head_descriptor}_model_optim_dict_model_idx_{model_idx}.tar"
            else:
                head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"

            loss_model_pth_path = os.path.join(self.loss_model_save_path, head_optim_save_file_name)
            if os.path.exists(loss_model_pth_path):
                # print(head_optim_save_file_name)
                if self.hvd or self.dist:
                    with FileLock(os.path.expanduser("~/.horovod_lock")):
                        chpt = torch.load(os.path.join(self.loss_model_save_path, head_optim_save_file_name),
                                          map_location='cpu'
                                          )
                        # model_state_dict, optim_state_dict = chpt["model"], chpt["optim"]
                else:
                    chpt = torch.load(os.path.join(self.loss_model_save_path, head_optim_save_file_name),
                                      map_location='cpu'
                                      )
                model_state_dict = chpt["model"]
            else:
                tmp_model = construct_conv1d_modules(
                    [self.rep_feat_dim // 2, pred_feat_dim], n_in=self.rep_feat_dim, last_act=False, bn=False
                ).cuda()
                model_state_dict = tmp_model.state_dict()
                if not (self.hvd or self.dist) or ((self.hvd and hvd.rank() == 0) or (self.dist and self.local_rank == 0)):
                    if self.hvd or self.dist:
                        with FileLock(os.path.expanduser("~/.horovod_lock")):
                            print(f"Saving {head_optim_save_file_name}. hvd = {self.hvd}")
                            # WE should save the weights in the memory for further loading...
                            head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"
                            head_states = {
                                "model": tmp_model.state_dict(),
                            }
                            torch.save(head_states, os.path.join(self.loss_model_save_path, head_optim_save_file_name))
                    else:
                        print(f"Saving {head_optim_save_file_name}. hvd = {self.hvd}")
                        # WE should save the weights in the memory for further loading...
                        head_optim_save_file_name = f"{head_descriptor}_model_optim_dict.tar"
                        head_states = {
                            "model": tmp_model.state_dict(),
                        }
                        torch.save(head_states, os.path.join(self.loss_model_save_path, head_optim_save_file_name))

            self.pred_feat_dim_to_head_list[i][cur_head_idx].load_state_dict(model_state_dict)

    ''' APPLY unary operation '''
    # todo: change unary operation's number to 8
    def apply_unary_operation(self, mtx, uop, feat_dim):
        # identity, square, 2, -, -2, inv, centralize
        if uop == IDENTITY:
            ans = mtx
        elif uop == SQUARE:
            ans = mtx ** 2
        elif uop == DOUBLE:
            ans = 2 * mtx
        elif uop == NEGATIVE:
            ans = -mtx
        # elif uop == 4:
        #     ans = -2 * mtx
        # no inverse operation here
        elif uop == INVERSE:
            with torch.no_grad():
                if len(feat_dim) == 1:
                    # todo: verify the effectiveness of the pinverse operation on [k x 1]-dim tensor --- what's the resulting tensor's dimension?
                    # Fature dimension remains not influenced
                    norm_2 = torch.norm(mtx, dim=-1, keepdim=True, p=2)
                    ans = mtx / torch.clamp(norm_2 ** 2, min=1e-9)
                    # ans = mtx.unsqueeze(-1)
                    # ans = torch.pinverse(ans)
                    # ans = ans.squeeze(-2)
                elif len(feat_dim) == 2:
                    # ans = torch.pinverse(mtx)
                    # ans = torch.linalg.pinv(mtx)
                    with torch.no_grad():
                        u, s, vh = np.linalg.svd(mtx.detach().cpu().numpy())
                    u = torch.from_numpy(u).cuda()
                    vh = torch.from_numpy(vh).cuda()
                    s = torch.from_numpy(s).cuda()

                    # s = torch.linalg.pinv(s)
                    s_norm_2 = torch.norm(s, dim=-1, keepdim=True, p=2)
                    s = s / torch.clamp(s_norm_2 ** 2, min=1e-9)
                    # ans = torch.matmul(torch.matmul(u, s), vh)
                    v = vh.contiguous().transpose(-1, -2).contiguous()
                    uh = u.contiguous().transpose(-1, -2).contiguous()
                    # ans = torch.matmul(v, torch.matmul(s, uh))
                    # print(v.size(), s.size(), uh.size()); s.size = bz x N x 3
                    # ans = torch.matmul(v * s.unsqueeze(-1), uh)
                    # right multiplication modifies column vectors
                    ans = torch.matmul(v * s.unsqueeze(-2), uh)
                    feat_dim = list(reversed(feat_dim))
                else:
                    raise ValueError(f"Illegal matrix feature dimension: {feat_dim}.")
        elif uop == ORTHOGONIZE:
            # ONLY feature space orthogonization
            if len(feat_dim) == 1:
                # then then orthogonal operation is reduced to normalization
                ans = torch.div(mtx, torch.clamp(torch.norm(mtx, dim=-1, p=2, keepdim=True), min=1e-9))
            elif len(feat_dim) == 2:
                with torch.no_grad():
                    u, s, vh = np.linalg.svd(mtx.detach().cpu().numpy())
                u = torch.from_numpy(u).cuda()
                vh = torch.from_numpy(vh).cuda()
                # todo: any other situation beyond these two?
                if len(u.size()) == 4:
                    u = u[:, :, :, :feat_dim[-1]]
                elif len(u.size()) == 5:
                    u = u[:, :, :, :, :feat_dim[-1]]
                else:
                    raise ValueError(f"Uncorrect u.size() = {u.size()}.")
                ans = torch.matmul(u, vh)
            else:
                raise ValueError(f"Uncorrect feat_dim.len = {len(feat_dim)}")
        elif uop == CENTRALIZE:
            if len(feat_dim) == 1 and len(mtx.size()) == 4:
                ans = mtx - torch.mean(mtx, dim=2, keepdim=True)
            else:
                ans = mtx
        else:
            raise ValueError(f"Unrecognized uop: {uop}")
        return ans, feat_dim

    ''' APPLY binary operation: 
     - We should consider the situation when the dimensions are not well aligned; 
     - We should have a function judging whether one sampled operation dictionary is valid or not...
     - Feature dimensions must be aligned, we only consider the situation when the k-dimension is not well aligned
    '''
    def apply_binary_operation(self, lft_mtx, rgt_mtx, lft_feat_dim, rgt_feat_dim, bop):
        # add, minus, element-wise multiply, cross product, cartesian product, matrix-vector product
        # If the sampled points' dimension is not aligned, expand the corresponding dimension
        # print(f"In apply_binary function with bop = {bop}, lft_mtx.size = {lft_mtx.size()}, rgt_mtx.size = {rgt_mtx.size()}, lft_feat_dim = {lft_feat_dim}, rgt_feat_dim = {rgt_feat_dim}.")
        if len(lft_mtx.size()) < len(rgt_mtx.size()):
            lft_mtx = lft_mtx.unsqueeze(2)
            if bop in [CROSS_PRODUCT, CARTESIAN_PRODUCT, MATRIX_VECTOR_PRODUCT]:
                k = rgt_mtx.size(2)
                if len(lft_mtx.size()) == 4:
                    lft_mtx = lft_mtx.repeat(1, 1, k, 1)
                elif len(lft_mtx.size()) == 5:
                    lft_mtx = lft_mtx.repeat(1, 1, k, 1, 1)
        elif len(lft_mtx.size()) > len(rgt_mtx.size()):
            rgt_mtx = rgt_mtx.unsqueeze(2)
            if bop in [CROSS_PRODUCT, CARTESIAN_PRODUCT, MATRIX_VECTOR_PRODUCT]:
                k = lft_mtx.size(2)
                if len(rgt_mtx.size()) == 4:
                    rgt_mtx = rgt_mtx.repeat(1, 1, k, 1)
                elif len(rgt_mtx.size()) == 5:
                    rgt_mtx = rgt_mtx.repeat(1, 1, k, 1, 1)

        if lft_mtx.size(2) < rgt_mtx.size(2):
            k = rgt_mtx.size(2)
            if len(lft_mtx.size()) == 4:
                lft_mtx = lft_mtx.repeat(1, 1, k, 1)
            elif len(lft_mtx.size()) == 5:
                lft_mtx = lft_mtx.repeat(1, 1, k, 1, 1)
        elif rgt_mtx.size(2) < lft_mtx.size(2):
            k = lft_mtx.size(2)
            if len(rgt_mtx.size()) == 4:
                rgt_mtx = rgt_mtx.repeat(1, 1, k, 1)
            elif len(rgt_mtx.size()) == 5:
                rgt_mtx = rgt_mtx.repeat(1, 1, k, 1, 1)

        # print(f"After reshaping matrices, lft_mtx.size = {lft_mtx.size()}, rgt_mtx.size = {rgt_mtx.size()}.")

        if bop == ADD:
            ans = lft_mtx + rgt_mtx
            feat_dim = lft_feat_dim
        elif bop == MINUS:
            ans = lft_mtx - rgt_mtx
            feat_dim = lft_feat_dim
        elif bop == MULTIPLY:
            ans = lft_mtx * rgt_mtx
            feat_dim = lft_feat_dim
        elif bop == CROSS_PRODUCT:
            ans = torch.cross(lft_mtx, rgt_mtx)
            feat_dim = lft_feat_dim
        elif bop == CARTESIAN_PRODUCT:
            # print(lft_mtx.size(), rgt_mtx.size(), lft_feat_dim, rgt_feat_dim)
            ans = torch.matmul(lft_mtx.unsqueeze(-1).contiguous(), rgt_mtx.unsqueeze(-2).contiguous())
            # print(f"In cartesian product operation with reshaped lft_mtx.size = {lft_mtx.unsqueeze(-1).size()}, rgt_mtx.size = {rgt_mtx.unsqueeze(-2).size()}, ans.size = {ans.size()}")

            feat_dim = [lft_feat_dim[0], rgt_feat_dim[0]]
        elif bop == MATRIX_VECTOR_PRODUCT:
            ans = torch.matmul(lft_mtx, rgt_mtx)
            feat_dim = [lft_feat_dim[0], rgt_feat_dim[1]]
        else:
            raise ValueError(f"Unrecognized bop: {bop}.")
        return ans, feat_dim

    ''' APPLY grouping operation '''
    def apply_grouping_operation(self, mtx, gop, feat_dim):
        # sum, mean, max, svd
        # print(mtx.size(), feat_dim, gop)
        assert 2 + len(feat_dim) <= len(mtx.size()), f"Dimension of the corresponding feature dose not satisfy the required condition: feat_dim = {feat_dim}, mtx.size() = {mtx.size()}."
        # print(mtx.size(), feat_dim, gop)
        if (2 + len(feat_dim) == len(mtx.size())):
            ans = mtx
        elif gop == SUM:
            ans = torch.sum(mtx, dim=2, keepdim=True)
        elif gop == MEAN:
            ans = torch.mean(mtx, dim=2, keepdim=True)
        elif gop == MAX:
            ans, _ = torch.max(mtx, dim=2, keepdim=True)
        elif gop == SVD:
            # Only the case: bz x N x k x 3
            if len(feat_dim) == 1 and 3 + len(feat_dim) == len(mtx.size()) and mtx.size(2) > 1:
                with torch.no_grad():
                    u, s, vh = np.linalg.svd(mtx.detach().cpu().numpy())
                ans = torch.from_numpy(vh).to(mtx.device) # .transpose(1, 2)
                feat_dim = [feat_dim[-1], feat_dim[-1]]
                ans = ans.contiguous().unsqueeze(2).contiguous()
            # elif 3 + len(feat_dim) == len(mtx.size()):
            #     # Then has one
            else:
                ans = mtx
        else:
            raise ValueError(f"Unrecognized gop: {gop}")
        return ans, feat_dim

    def from_oper_to_one_level_mtx(self, mtxs, oper_idx):
        ans = mtxs[0]
        if len(mtxs) > 1:
            if oper_idx == 0:
                ans = mtxs[0]
            elif oper_idx == 1:
                ans = mtxs[1]
            elif oper_idx == 2:
                ans = mtxs[0] * mtxs[1]
            elif oper_idx == 3:
                ans = mtxs[0] + mtxs[1]
            elif oper_idx == 4:
                ans = mtxs[0] - mtxs[1]
            elif oper_idx == 5:
                ans = mtxs[1] - mtxs[0]
            elif oper_idx == 6:
                ans = torch.cross(mtxs[0], mtxs[1])
            else:
                raise ValueError(f"Unrecognized oper_idx: {oper_idx}.")
        else:
            if oper_idx == 0:
                ans = mtxs[0]
            elif oper_idx == 1:
                ans = mtxs[0] * 2
            elif oper_idx == 2:
                ans = mtxs[0] ** 2
            elif oper_idx == 3:
                ans = -1 * mtxs[0]
            else:
                raise ValueError(f"Unrecognized oper_idx: {oper_idx}.")
        return ans

    ''' CALCULATE to-pred-feats by sampled operation dictionary '''
    def generate_gt_fats_by_oper_dict(self, oper_dict, mtxs):
        # pos, feat = mtxs
        if "oper" in oper_dict:
            oper_idx = oper_dict["oper"]
            # assert  oper_idx < len(mtxs), f"Number of matrices = {len(mtxs)}, oper_idx = {oper_idx}."
            # ans = mtxs[oper_idx]
            ans = self.from_oper_to_one_level_mtx(mtxs, oper_idx)
            uop = oper_dict["uop"] # , oper_dict["gop"]
            feat_dim = [self.in_feat_dim]
            ans, feat_dim = self.apply_unary_operation(ans, uop, feat_dim)
        elif "chd" in oper_dict:
            uop, gop = oper_dict["uop"], oper_dict["gop"]
            chd_oper_dict = oper_dict["chd"]
            chd_oper_ans, chd_feat_dim = self.generate_gt_fats_by_oper_dict(chd_oper_dict, mtxs)
            # if chd_oper_ans is None
            assert chd_oper_ans is not None
            ans, feat_dim = self.apply_unary_operation(chd_oper_ans, uop, chd_feat_dim)
            ans, feat_dim = self.apply_grouping_operation(ans, gop, feat_dim)
        else:
            bop, gop = oper_dict["bop"], oper_dict["gop"]
            uop = oper_dict["uop"]
            lft_chd_oper_dict, rgt_chd_oper_dict = oper_dict["lft_chd"], oper_dict["rgt_chd"]
            lft_chd_ans, lft_chd_feat_dim  = self.generate_gt_fats_by_oper_dict(lft_chd_oper_dict, mtxs)
            rgt_chd_ans, rgt_chd_feat_dim = self.generate_gt_fats_by_oper_dict(rgt_chd_oper_dict, mtxs)
            bop = (bop - 1) % self.nn_binary_opers
            ans, feat_dim = self.apply_binary_operation(lft_chd_ans, rgt_chd_ans, lft_chd_feat_dim, rgt_chd_feat_dim, bop)
            # print(f"bop after bop: {ans.size()}")
            ans, feat_dim = self.apply_unary_operation(ans, uop, feat_dim)
            # print(f"uop after uop: {ans.size()}")
            ans, feat_dim = self.apply_grouping_operation(ans, gop, feat_dim)
            # print(f"gop after gop: {ans.size()}")

        return ans, feat_dim

    ### Use this function when there is no normal input
    ''' ESTIMATE normal vectors for each point '''
    def estimate_normals(self, pos):
        # pos.size = bz x N x 3
        normals = []
        for i in range(pos.size(0)):
            pts = pos[i].detach().cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            nms = np.array(pcd.normals)
            normals.append(torch.from_numpy(nms).to(pos.device).float().unsqueeze(0))
        normals = torch.cat(normals, dim=0)
        return normals

    ''' GET position offset vector for each point from the center position of the part belonged to --- another feature acturally '''
    def calculate_part_offset(self, pos, labels):
        # normal is a part-level feature; such information is also a part-level feature...
        # labels for each point
        # bz x N x 3
        # bz x N x nmasks
        # bz x nmasks x N x 3
        masked_mult_pos = labels.contiguous().transpose(1, 2).unsqueeze(-1) * pos.unsqueeze(1)
        transposed_masks = labels.contiguous().transpose(1, 2).contiguous()
        # bz x nmasks x 3 = bz x nmasks x 3 / bz x nmasks x 1
        masked_mult_pos = torch.sum(masked_mult_pos, dim=2) / torch.clamp(torch.sum(transposed_masks, dim=-1).unsqueeze(-1), min=1e-9)
        _, seg_labels = torch.max(labels, dim=-1)
        # bz x N x 3
        selected_center_pos = batched_index_select(values=masked_mult_pos, indices=seg_labels, dim=1)
        offset = selected_center_pos - pos
        return offset
        # pass

    ''' GET operator A & operatior B by operator index '''
    # point-level operator selection
    def get_oper_matrix_from_oper(self, oper_idx, oper_matrix_a, oper_matrix_b):
        ans = None
        if oper_idx == 0:  # matrix a
            ans = oper_matrix_a
        elif oper_idx == 1:  # matrix b
            ans = oper_matrix_b
        elif oper_idx == 2:  # square
            ans = oper_matrix_a ** 2
        elif oper_idx == 3:  # square
            ans = oper_matrix_b ** 2
        elif oper_idx == 4:  # element-wise product
            if oper_matrix_a.size(-1) == oper_matrix_b.size(-1):
                ans = oper_matrix_a * oper_matrix_b
            else:
                ans = oper_matrix_a
        elif oper_idx == 5:  # cross product
            if oper_matrix_a.size(-1) == oper_matrix_b.size(-1):
                ans = torch.cross(oper_matrix_a, oper_matrix_b)
            else:
                ans = oper_matrix_a
        else:
            raise ValueError(f"Unrecognized operator idx: {oper_idx}.")
        return ans

    ''' TRANSFORM point-level feature matrix by transformation index '''
    # point-level feature transformation
    def trans_oper_matrix_by_oper_trans_idx(self, oper_matrix, oper_trans_idx, center_mtx=None):
        ans = None
        if oper_trans_idx == 0:
            ans = oper_matrix
        elif oper_trans_idx == 1:
            ans = -oper_matrix
        elif oper_trans_idx == 2:
            ans = 2 * oper_matrix
        elif oper_trans_idx == 3:
            avg_oper_matrix = torch.mean(oper_matrix, dim=2, keepdim=True)
            ans = oper_matrix - avg_oper_matrix
        elif oper_trans_idx == 4:
            if center_mtx is None:
                avg_oper_matrix = torch.mean(oper_matrix, dim=2, keepdim=True)
                ans = oper_matrix - avg_oper_matrix
            else:
                ans = oper_matrix - center_mtx.unsqueeze(2)
        else:
            raise ValueError(f"Unrecognized oper_trans_idx: {oper_trans_idx}.")
        return ans

    ''' NOT using now '''
    ''' GET part-level feature matrix by part-level operator index '''
    def get_pred_feat_matrix_by_oper_matrix_idx(self, oper_mtxs, part_oper_idx):
        ans = None
        if part_oper_idx == 0:
            left_oper, right_oper = oper_mtxs
            bz, N, k = left_oper.size(0), left_oper.size(1), left_oper.size(2)
            mtx = torch.matmul(left_oper.contiguous().transpose(2, 3).contiguous(), right_oper)
            ans = mtx.contiguous().view(bz, N, -1)
        elif part_oper_idx == 1:
            left_oper, right_oper = oper_mtxs
            bz, N, k = left_oper.size(0), left_oper.size(1), left_oper.size(2)
            if not self.no_spectral:
                try:
                    mtx = torch.matmul(left_oper.contiguous().transpose(2, 3).contiguous(), right_oper)
                    U, _, VT = np.linalg.svd(mtx.detach().cpu().numpy())
                    U, VT = torch.from_numpy(U).to(left_oper.device), torch.from_numpy(VT).to(left_oper.device)
                    mtx = torch.matmul(U, VT)
                except:
                    mtx = torch.matmul(left_oper.contiguous().transpose(2, 3).contiguous(), right_oper)
            else:
                mtx = torch.matmul(left_oper.contiguous().transpose(2, 3).contiguous(), right_oper)
            if torch.any(torch.isnan(mtx)) or torch.any(torch.isinf(mtx)):
                ans = None
            else:
                ans = mtx.contiguous().view(bz, N, -1)
        elif part_oper_idx == 2:

            A = oper_mtxs[0]
            b = torch.sum(oper_mtxs[1], dim=-1, keepdim=True)
            bz, N = A.size(0), A.size(1)
            if not self.no_spectral:
                with torch.no_grad():
                    x = torch.linalg.lstsq(A, b)[0]
                x = x.squeeze(-1)

                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    ans = None
                else:
                    ans = x
            else:
                ans = torch.matmul(A.contiguous().transpose(2, 3).contiguous(), b)
                ans = ans.contiguous().view(bz, N, -1).contiguous()
        elif part_oper_idx == 3:
            oper = oper_mtxs[0]
            # s.size = bz x N x 3
            with torch.no_grad():
                u, s, vh = np.linalg.svd(oper.detach().cpu().numpy())
            v = torch.from_numpy(vh).to(oper.device).transpose(1, 2)
            # min_s.size = bz x N; min_s_idx.size = bz x N;
            min_s, min_s_idx = torch.min(torch.from_numpy(s).to(oper.device), dim=-1)
            rt = v[:, :, min_s_idx]
            if torch.any(torch.isnan(rt)) or torch.any(torch.isinf(rt)):
                ans = None
            else:
                ans = rt
        elif part_oper_idx == 4:
            # A.size = bz x N x k x k_1
            # b.size = bz x N x k x k_2
            A = oper_mtxs[0]
            b = oper_mtxs[1]
            bz, N = A.size(0), A.size(1)
            if not self.no_spectral:
                with torch.no_grad():
                    x = torch.linalg.lstsq(A, b)[0]
                x = x.squeeze(-1)

                if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                    ans = None
                else:
                    ans = x.contiguous().view(bz, N, -1).contiguous()
            else:
                ans = torch.matmul(A.contiguous().transpose(2, 3).contiguous(), b)
                ans = ans.contiguous().view(bz, N, -1).contiguous()
            # ans = x
        elif part_oper_idx == 5:  # Element-wise MEAN of a point-level feature matrix
            A = oper_mtxs[0]
            ans = torch.mean(A, dim=-2)
        elif part_oper_idx == 6: # Element-wise SUM of a point-level feature matrix
            A = oper_mtxs[0]
            ans = torch.sum(A, dim=-2)
        elif part_oper_idx == 7: # (1) Inner product between each two points; (2) MEAN of the inner product value
            A, B = oper_mtxs[0], oper_mtxs[1]
            if A.size(-1) != B.size(-1):
                ans = torch.mean(torch.sum(A ** 2, dim=-1), dim=-1) + torch.mean(torch.sum(B ** 2, dim=-1), dim=-1)
            else:
                inner_prod = torch.matmul(A, B.contiguous().transpose(2, 3).contiguous())
                ans = torch.mean(torch.mean(inner_prod, dim=-1), dim=-1)
        elif part_oper_idx == 8: # (1) Cross product between each two points; (2) MEAN over the cross product vectors
            A, B = oper_mtxs[0], oper_mtxs[1]
            k, feat_dim = A.size(-2), A.size(-1)
            if A.size(-1) != B.size(-1) or A.size(-2) != B.size(-2):
                ans = torch.mean(A, dim=-2)
            else:
                # A.size = bz x N x k x 3
                A_expand = A.unsqueeze(-2).repeat(1, 1, 1, k, 1)
                B_expand = B.unsqueeze(-3).repeat(1, 1, k, 1, 1)
                cross_prod = torch.cross(A_expand, B_expand, dim=-1)
                ans = torch.mean(torch.mean(cross_prod, dim=-2), dim=-2)

        else:
            raise ValueError(f"Unrecognized part_oper_idx: {part_oper_idx}.")
        return ans

    ''' NOT using now '''
    ''' GET matmul matrix ans of different kinds 
     - This is for loss model v2'''
    def get_matmul_matrix_transformation_res(self, mtx, transform_idx):
        if transform_idx == 0:
            ans = mtx
        elif transform_idx == 1: # pinverse
            ans = torch.pinverse(mtx)
        elif transform_idx == 2:
            U, _, VT = np.linalg.svd(mtx.detach().cpu().numpy())
            U, VT = torch.from_numpy(U).to(mtx.device), torch.from_numpy(VT).to(mtx.device)
            feat_dim = VT.size(-1)
            U = U[:, :, :, :feat_dim]
            ans = torch.matmul(U, VT)
        else:
            raise ValueError(f"Unrecognzied mtx transformation index for matmul operation: {transform_idx}.")
        return ans

    ''' GET neighbouring pts for each point 
     - Sample neighbours only based on labels'''
    def get_near_sim_part_neighbours(self, label, pos, r=0.07):
        with torch.no_grad():
            # pp_label_sim.size = bz x N x N
            pp_label_sim = torch.matmul(label, label.transpose(1, 2)) / \
                           torch.clamp(torch.norm(label, dim=2, keepdim=True) * torch.norm(label.transpose(1, 2), dim=1,
                                                                                           keepdim=True), min=1e-9)

            pp_sampling_dist = pp_label_sim / torch.clamp(torch.sum(pp_label_sim, dim=-1, keepdim=True), min=1e-9)
            bz, N = pp_sampling_dist.size(0), pp_sampling_dist.size(1)
            pp_sampling_dist_expand = pp_sampling_dist.contiguous().view(bz * N, -1).contiguous()
            sampled_neis_expand = torch.multinomial(pp_sampling_dist_expand, self.pp_sim_k, replacement=True)
            sampled_neis = sampled_neis_expand.contiguous().view(bz, N, -1).contiguous()
        return sampled_neis

    def forward(
            self,
            pos,
            x,
            label,
            matrices={},
            oper_dict=[],
            return_predicted_feat=False
        ):
        bz, N, x_k = pos.size(0), pos.size(1), x.size(-1)

        ''' ESTIMATE normals '''
        if 'normals' not in matrices:
            if 'flow' in matrices:
                normals = matrices['flow']
            else:
                normals = self.estimate_normals(pos)
            # normals = self.calculate_part_offset(pos, label)
        else:

            normals = matrices['normals']
        ''' ESTIMATE normals '''

        ''' INITIALIZE loss '''
        inter_gt_l = torch.zeros((1,), dtype=torch.float32, device=x.device)
        ''' INITIALIZE loss '''

        if oper_dict is None:
            return inter_gt_l

        predicted_feat = []

        for i, cur_oper_dict in enumerate(oper_dict):
            # GET part-level operator index

            # A valid operator index
            if self.module_index_to_selected_oper_dict[i] is not None:

                # cur_oper_dict_list = self.from_oper_dict_to_oper_list(cur_oper_dict)
                # module_index = self.selected_oper_dict_to_module_index[cur_oper_dict_list]
                ''' GENERATE part-level features for each point '''
                with torch.no_grad(): # label.size = bz x N x k
                    if self.args.part_aware:
                        pp_label_sim = torch.matmul(label, label.transpose(1, 2)) / \
                                       torch.clamp(torch.norm(label, dim=2, keepdim=True) * torch.norm(label.transpose(1, 2), dim=1,
                                                                                                       keepdim=True), min=1e-9)
                    else:
                        pp_label_sim = torch.matmul(label, label.transpose(1, 2)) / \
                                       torch.clamp(
                                           torch.norm(label, dim=2, keepdim=True) * torch.norm(label.transpose(1, 2),
                                                                                               dim=1,
                                                                                               keepdim=True), min=1e-9)
                        pp_label_sim[:, :, :] = 1.

                    if self.args.restrict_nei:
                        # r = 0.5
                        r = 0.5
                        pp_pos_dist = torch.sqrt(torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1))
                        pp_label_sim[pp_pos_dist > r] = 0.0

                    # pp_label_sim = pp_label_sim.cuda()

                    permidx = np.random.permutation(N)
                    permidx = torch.from_numpy(permidx).to(pos.device)
                    pp_label_sim = pp_label_sim[:, :, permidx]
                    top_k_label_sim, top_k_label_sim_idx = torch.topk(pp_label_sim, k=self.pp_sim_k, dim=-1, largest=True)
                    perm_pos = pos[:, permidx]
                    perm_normals = normals[:, permidx]
                    if not self.args.geometric_aware:
                        perm_labels = label[:, permidx].float()


                # permidx = np.random.permutation(N)
                # permidx = torch.from_numpy(permidx).to(pos.device)
                # pp_label_sim = pp_label_sim[:, :, permidx]
                # top_k_label_sim, top_k_label_sim_idx = torch.topk(pp_label_sim, k=self.pp_sim_k, dim=-1, largest=True)
                # perm_pos = pos[:, permidx]
                # perm_normals = normals[:, permidx]
                pp_part_pos = batched_index_select(values=perm_pos, indices=top_k_label_sim_idx, dim=1)
                pp_part_normals = batched_index_select(values=perm_normals, indices=top_k_label_sim_idx, dim=1)

                if not self.args.geometric_aware:
                    pp_part_labels = batched_index_select(values=perm_labels, indices=top_k_label_sim_idx, dim=1)

                # ''' GENERATE part-level features for each point '''
                # # get samled neis
                # sampled_neis = self.get_near_sim_part_neighbours(label, pos)
                # pp_part_pos = batched_index_select(values=pos, indices=sampled_neis, dim=1)
                # pp_part_normals = batched_index_select(values=normals, indices=sampled_neis, dim=1)
                #
                # ''' GENERATE part-level features for each point '''

                if not self.args.geometric_aware:
                    to_pred_feat, to_pred_feat_dim = self.generate_gt_fats_by_oper_dict(cur_oper_dict,
                                                                                        [pp_part_labels])
                else:
                    to_pred_feat, to_pred_feat_dim = self.generate_gt_fats_by_oper_dict(cur_oper_dict, [pp_part_pos, pp_part_normals])

                if to_pred_feat is None:
                    continue

                if len(to_pred_feat_dim) + 2 < len(to_pred_feat.size()):
                    to_pred_feat = to_pred_feat.contiguous().squeeze(2).contiguous()

                if len(to_pred_feat_dim) > 1:
                    # bz, N =
                    to_pred_feat = to_pred_feat.contiguous().view(bz, N, -1).contiguous()

                cur_head_idx = self.module_index_to_head_idx_dict[i]

                if to_pred_feat is not None and not (torch.any(torch.isnan(to_pred_feat))):
                    ''' CALCULATE predicted features & mse loss '''
                    pred_feat = CorrFlowPredNet.apply_module_with_conv1d_bn(
                        x,
                        self.pred_feat_dim_to_head_list[i][cur_head_idx]
                    )
                    # print(to_pred_feat.size(), pred_feat.size())
                    # GET l2-loss between predicted features and ground-truth prediction features
                    cur_loss = torch.mean(torch.sum((pred_feat - to_pred_feat) ** 2, dim=-1))
                    ''' CALCULATE predicted features & mse loss '''
                    inter_gt_l += cur_loss
                    predicted_feat.append(pred_feat)

        if return_predicted_feat:
            if len(predicted_feat) > 0:
                predicted_feat = torch.cat(predicted_feat, dim=-1)
            return inter_gt_l, predicted_feat

        return inter_gt_l