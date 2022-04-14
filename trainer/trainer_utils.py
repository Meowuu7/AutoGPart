from typing import Dict, Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from model.utils import batched_index_select, hungarian_matching
from model.model_util import init_weight
import numpy as np
SQRT_EPS = 0.000001 # (1e-5)
DIVISION_EPS = 0.000001 # (1e-5)
from math import pi as PI
from model.loss_model_v5 import ComputingGraphLossModel as loss_model

def construct_linear_model(mlp_dims, in_dim):
    module_list = []
    in_dims, out_dims = [in_dim] + mlp_dims[:-1], mlp_dims
    for i, (ind, oud) in enumerate(zip(in_dims, out_dims)):
        if i < len(in_dims) - 1:
            module_list.append(nn.Linear(ind, oud, bias=True))
            module_list.append(nn.ReLU())
        else:
            module_list.append(nn.Linear(ind, oud))
    constructed_modules = nn.Sequential(*module_list)
    init_weight(constructed_modules)
    return constructed_modules


def get_masks_for_seg_labels(seg_labels, maxx_label=None):
    # print(seg_labels)
    # print(torch.max(seg_labels))
    if maxx_label is None:
        maxx_label = int(torch.max(seg_labels).detach().item())
    ### SET invalid instance idx to the last instance idx ###
    # seg_labels[seg_labels == -1] = maxx_label
    assert not torch.any(seg_labels == -1)
    masks = np.eye(maxx_label + 1)[np.minimum(seg_labels.detach().cpu().numpy(), maxx_label).astype('int32')]
    masks = torch.from_numpy(masks).to(seg_labels.device).float()
    return masks


def get_current_lr(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    return lr


def set_lr_by_value(optimizer, tgt_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = tgt_lr


def clamp_matrix(mat):
    mat = torch.clamp(mat, min=-1e9, max=1e9)
    return mat


def acos_safe(x):
    # if torch.any(torch.isnan(x)):
    #     print("x in acos_safe has nan value!")
    return torch.acos(torch.clamp(x, min=(-1.0 + (1e-5)), max=(1.0 - (1e-5))))
    # return torch.acos(torch.clamp(x, min=(-1.0), max=(1.0)))


def sqrt_safe(x):
    return torch.sqrt(torch.abs(x) + SQRT_EPS)

def compute_param_loss(pred, T_gt, T_param_gt):
    '''
    only add loss to corresponding type
    pred: (B, N, 22)
    T_gt: (B, N)
    T_param_gt: (B, N, 22)
    '''
    param_list = {5:[0,4], 1:[4,8], 4:[8,15], 3:[15,22]}

    #[0, 4, 8, 15, 22]

    b, N, _ = pred.shape

    #l2_loss = nn.MSELoss(reduction='sum')
    l2_loss = nn.MSELoss()

    total_loss = 0
    length = 0
    cnt = 0
    avg_valid_ratio = []
    for b in range(pred.shape[0]):
        for i in [1, 4, 5, 3]:
            index = T_gt[b] == i
            tmp_pred = pred[b][index]
            tmp_gt = T_param_gt[b][index]

            if tmp_pred.shape[0] == 0:
                continue
            if tmp_gt.sum() == 0: # no parameters to process
                continue

            tmp_pred = tmp_pred[:, param_list[i][0]:param_list[i][1]]
            tmp_gt = tmp_gt[:, param_list[i][0]:param_list[i][1]].float()

            tot_shape_nn = tmp_gt.shape[0]

            valid_mask = tmp_gt.sum(1) != 0

            tmp_pred = tmp_pred[valid_mask]
            tmp_gt = tmp_gt[valid_mask]

            filtered_shape_nn = tmp_gt.shape[0]

            avg_valid_ratio.append(float(filtered_shape_nn) / float(tot_shape_nn))

            if tmp_gt.shape[0] == 0:
                continue

            tmp_loss = l2_loss(tmp_pred, tmp_gt)

            # ignore wrong type label
            if tmp_gt.max() > 10 or tmp_loss > 50:
                continue

            total_loss += tmp_loss

            length += tmp_pred.shape[0]
            cnt += 1

    #TODO: only happened in test phase
    if cnt == 0:
        if torch.isnan(l2_loss(tmp_pred, tmp_gt.float())).sum() > 0:
            return torch.Tensor([0.0]).to(T_gt.device)
        return l2_loss(tmp_pred, tmp_gt.float())

    total_loss = total_loss / cnt

    # print(f"valid ratio = {sum(avg_valid_ratio) / len(avg_valid_ratio)}")

    return total_loss


class MultiNominalWithParamsDistribution(nn.Module):
    def __init__(self, mu_k1=1.0, sigma_k1=0.2, n_layers=2, device=None):
        super(MultiNominalWithParamsDistribution, self).__init__()

        self.mu_k1, self.sigma_k1 = mu_k1, sigma_k1
        self.n_layers = n_layers
        self.device = device

        self.mu_k1 = torch.ones((1, ), dtype=torch.float32, requires_grad=True).cuda() * self.mu_k1
        self.sigma_k1 = torch.ones((1, ), dtype=torch.float32, requires_grad=True).cuda() * self.sigma_k1

        self.mu_sigma_k2_net = construct_linear_model([16, 32, 2], in_dim=1)
        self.mu_sigma_k3_net = construct_linear_model([16, 32, 2], in_dim=2) # k1, k2
        self.mu_sigma_k4_net = construct_linear_model([16, 32, 2], in_dim=3) # k1, k2, k3
        # now we have [k1, k2, k3, k4] --- four one level parameters to construct [k1 * A + k2 * B, k3 * A + k4 * B]
        self.multinomial_params_net = construct_linear_model([16, 32, 4], in_dim=4) # input: [k1, k2, k3, k4] output: four params for multinomial dist
        self.level_2_dist = None

    def get_params(self):
        params = [self.mu_k1, self.sigma_k1, self.mu_sigma_k2_net]
        for sub_net in [self.mu_sigma_k2_net, self.mu_sigma_k3_net, self.mu_sigma_k4_net, self.multinomial_params_net]:
            for layer in sub_net:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    if layer.bias is not None:
                        params.append(layer.bias)
        return params

    def get_baseline_values(self):
        with torch.no_grad():
            k1 = self.mu_k1
            mu_sigma_k2 = self.mu_sigma_k2_net(k1)
            k2, _ = mu_sigma_k2[0], mu_sigma_k2[1]
            mu_sigma_k3 = self.mu_sigma_k3_net(
                torch.cat([k1, k2], dim=-1)
            )
            k3, _ = mu_sigma_k3[0], mu_sigma_k3[1]
            mu_sigma_k4 = self.mu_sigma_k4_net(
                torch.cat([k1, k2, k3], dim=-1)
            )
            k4, _ = mu_sigma_k4[0], mu_sigma_k4[1]
            level_2_multinomial_params = self.multinomial_params_net(
                torch.cat([k1, k2, k3, k4], dim=-1)
            )
            value = torch.argmax(level_2_multinomial_params, dim=-1)
            return value

    def forward(self):
        noise = torch.randn_like(self.mu_k1, requires_grad=False)
        k1 = self.mu_k1 + self.sigma_k1 * noise
        mu_sigma_k2 = self.mu_sigma_k2_net(k1)
        mu_k2, sigma_k2 = mu_sigma_k2[0], mu_sigma_k2[1]
        noise_k2 = torch.randn_like(mu_k2, requires_grad=False)
        k2 = mu_k2 + sigma_k2 * noise_k2
        mu_sigma_k3 = self.mu_sigma_k3_net(
            torch.cat([k1, k2], dim=-1)
        )
        mu_k3, sigma_k3 = mu_sigma_k3[0], mu_sigma_k3[1]
        noise_k3 = torch.randn_like(mu_k3, requires_grad=False)
        k3 = mu_k3 + sigma_k3 * noise_k3
        mu_sigma_k4 = self.mu_sigma_k4_net(
            torch.cat([k1, k2, k3], dim=-1)
        )
        mu_k4, sigma_k4 = mu_sigma_k4[0], mu_sigma_k4[1]
        noise_k4 = torch.randn_like(mu_k4, requires_grad=False)
        k4 = mu_k4 + sigma_k4 * noise_k4
        level_2_multinomial_params = self.multinomial_params_net(
            torch.cat([k1, k2, k3, k4], dim=-1)
        )
        level_2_multinomial_params = torch.sigmoid(level_2_multinomial_params)
        level_2_multinomial_params = level_2_multinomial_params / torch.clamp(level_2_multinomial_params.sum(), min=1e-9)
        self.level_2_dist = Categorical(level_2_multinomial_params.unsqueeze(0))
        values = self.level_2_dist.sample()
        return values

    def rein_update_params(self, values, rewards, baseline, lr):
        loss = 0.0
        n_samples = values.size(0)
        for i in range(n_samples):
            cur_sample, cur_rw = values[i], rewards[i]
            loss += (-self.level_2_dist.log_prob(cur_sample)) * (cur_rw - baseline)
        # print(loss.size())
        loss = loss.sum()
        loss /= n_samples

        loss.backward()

        params = self.get_params()
        grads = torch.autograd.grad(loss, params)
        for i, (param, grad) in enumerate(zip(params, grads)):
            param -= lr * grad
        return

#### Multi-multinomial distribution class ####
class MultiMultinomialDistribution(nn.Module):
    def __init__(self, n_dists, n_params, device=None):
        super(MultiMultinomialDistribution, self).__init__()
        self.n_dists = n_dists
        self.n_params = n_params
        self.device = device
        self.loss_accum = 0.0
        self.n_samples_accum = 0

        # CLAMP parameter values to a valid range and normalize the parameter
        self.params = torch.ones((self.n_dists, self.n_params), dtype=torch.float32, requires_grad=True) # .cuda()
        # with torch.no_grad():
        #     self.params[self.n_dists - 1, 1] = 0.0
        #     self.params[self.n_dists - 2, 1] = 0.0
        # self.params.requires_grad = True
        self.params = self.params / torch.sum(self.params, dim=-1, keepdim=True)

        ''' INITIALLY the categorical distribution '''
        self.cat_dist = Categorical(self.params)

    def load_params(self, preload_params):
        self.params = preload_params.clone()
        self.cat_dist = Categorical(self.params)

    ''' SAMPLE value from the distribution '''
    def sample_value(self):
        self.cat_dist = Categorical(self.params)
        values = self.cat_dist.sample()
        # if values[self.n_dists - 1].detach().item() == 1:
        #     values[self.n_dists - 1] = 0 if self.params[self.n_dists - 1, 0].detach().item() > self.params[self.n_dists - 1, 2].detach().item() else 2
        # if values[self.n_dists - 2].detach().item() == 1:
        #     values[self.n_dists - 2] = 0 if self.params[self.n_dists - 2, 0].detach().item() > self.params[self.n_dists - 2, 2].detach().item() else 2
        try:
            self.params.retain_grad()
        except:
            pass
        return values

    def get_baseline_values(self):
        baseline_values = torch.argmax(self.params, dim=-1)
        return baseline_values

    def rein_update_params(self, values, rewards, baseline, lr):
        # # values.size = n_samples, ndists x nparams
        # # rewards, baseline & lr: real values
        # loss = -(self.cat_dist.log_prob(values).unsqueeze(0)) * (rewards.unsqueeze(1).unsqueeze(-1) - baseline)
        loss = 0.0
        # self.cat_dist = Categorical(self.params)

        n_samples = values.size(0)
        for i in range(n_samples):
            cur_sample, cur_rw = values[i], rewards[i]
            loss += (-self.cat_dist.log_prob(cur_sample)) * (cur_rw - baseline)
        # print(loss.size())
        loss = loss.sum()
        loss /= n_samples

        grads = torch.autograd.grad(loss, self.params)
        print(grads)
        # loss.backward()
        self.params = self.params - grads[0] # self.params.grad
        with torch.no_grad():
            self.params = torch.clamp(self.params, min=0.0, max=1.0)
            # self.params[self.n_dists - 1, 1] = 0.0 # never sample 1 in ...
            # self.params[self.n_dists - 2, 1] = 0.0 # never sample 1 in ...
            self.params = self.params / torch.sum(self.params, dim=1, keepdim=True)

        self.params.requires_grad = True
        # loss = -self.cat_dist.log_prob(values) * (rewards - baseline)

    # todo: How to set proper n_samples, rewards, values, baselines...
    def rein_update_params_one_sample(self, values, rewards, baseline, lr, n_samples=1):
        # # values.size = n_samples, ndists x nparams
        # # rewards, baseline & lr: real values
        # loss = -(self.cat_dist.log_prob(values).unsqueeze(0)) * (rewards.unsqueeze(1).unsqueeze(-1) - baseline)
        loss = 0.0
        # self.cat_dist = Categorical(self.params)

        n_samples = values.size(0)

        self.cat_dist = Categorical(self.params)

        # for i in range(n_samples):
        # cur_sample, cur_rw = values[i], rewards[i]
        cur_sample, cur_rw = values, rewards
        loss += (-self.cat_dist.log_prob(cur_sample)) * (cur_rw - baseline)
        # print(loss.size())
        # loss = loss.sum()
        loss /= n_samples

        grads = torch.autograd.grad(loss, self.params)
        # print(grads)
        # loss.backward()
        self.params = self.params - grads[0] # self.params.grad
        with torch.no_grad():
            self.params = torch.clamp(self.params, min=0.0, max=1.0)
            # self.params[self.n_dists - 1, 1] = 0.0 # never sample 1 in ...
            # self.params[self.n_dists - 2, 1] = 0.0 # never sample 1 in ...
            self.params = self.params / torch.sum(self.params, dim=1, keepdim=True)

        self.params.requires_grad = True
        # loss = -self.cat_dist.log_prob(values) * (rewards - baseline)

    def reset_cat_dists_loss_n_samples(self):
        self.loss_accum = 0.0
        self.n_samples_accum = 0
        self.cat_dist = Categorical(self.params)

    ''' ACCUMULATE sampling loss by one single sample '''
    def accumulate_sampling_losses_one_sample(self, value, reward, baseline):
        cur_loss = (-self.cat_dist.log_prob(value)) * (reward - baseline)
        self.loss_accum += cur_loss
        self.n_samples_accum += 1

    ''' UPDATE parameters by accumulated losses '''
    def update_params_by_accum_loss(self, lr=None, n_samples=None, forbid_dict=None, preference_dict=None):
        if self.n_samples_accum == 0:
            return

        cur_nsamples = n_samples if n_samples is not None else self.n_samples_accum
        self.loss_accum /= cur_nsamples # self.n_samples_accum

        grads = torch.autograd.grad(self.loss_accum, self.params)
        self.params = self.params - grads[0]

        with torch.no_grad():
            self.params = torch.clamp(self.params, min=0.0, max=1.0)

            idx_to_possi = {}
            forbid_idx_to_possi = {}
            preference_idx_to_possi = {}
            normal_possi_accum = 1.0
            tot_normal_possi = 0.0
            for i in range(self.params.size(1)):
                cur_possi = self.params[0][i].detach().item()
                if forbid_dict is not None and i in forbid_dict:
                    forbid_idx_to_possi[i] = 0.0
                elif preference_dict is not None and i in preference_dict:
                    preference_idx_to_possi[i] = max(preference_dict[i], cur_possi)
                    normal_possi_accum -= preference_idx_to_possi[i]
                else:
                    idx_to_possi[i] = cur_possi
                    tot_normal_possi += cur_possi
            for normal_idx in idx_to_possi:
                idx_to_possi[normal_idx] = (idx_to_possi[normal_idx] / tot_normal_possi) * normal_possi_accum
            cur_params = []
            for i in range(self.params.size(1)):
                if i in idx_to_possi:
                    cur_params.append(idx_to_possi[i])
                elif i in preference_idx_to_possi:
                    cur_params.append(preference_idx_to_possi[i])
                elif i in forbid_idx_to_possi:
                    cur_params.append(0.0)
                else:
                    raise ValueError(f"Not in any dict: {i}!")
            self.params = torch.tensor(cur_params, dtype=torch.float32).unsqueeze(0) # .cuda()

            # self.params = self.params / torch.sum(self.params, dim=1, keepdim=True)

        self.params.requires_grad = True

        self.reset_cat_dists_loss_n_samples()

class DistributionTreeNodeArch:
    # convolutional modules can be chosen from [PointNet++ module, EdgeConv module, PointNet++ without up-conv module, Skip]
    def __init__(self, cur_depth=0, tot_layers=3, nn_conv_modules=4, device=None, args=None, preload_params=None):
        '''
        :param cur_depth:
        :param tot_layers:
        :param nn_conv_modules: idx for conv_module --- 0: PointnetPP module, 1: EC module; 2: PointNetPP_v2 module; 3: Skip
        :param device:
        :param args:
        '''
        self.args = args
        self.depth = cur_depth
        self.tot_layers = tot_layers
        self.nn_conv_modules = nn_conv_modules
        self.device = device

        self.arch_dist = MultiMultinomialDistribution(n_dists=1, n_params=nn_conv_modules, device=device) # .cuda()
        if preload_params is not None:
            self.arch_dist.load_params(torch.from_numpy(preload_params["param_dist"]))

        if cur_depth < tot_layers - 1:
            self.children = []
            for i_chd in range(nn_conv_modules):
                cur_child = DistributionTreeNodeArch(cur_depth=self.depth + 1, tot_layers=self.tot_layers, nn_conv_modules=self.nn_conv_modules, device=self.device, args=self.args, preload_params=preload_params[f"chd_{i_chd}"] if preload_params is not None else None)
                self.children.append(cur_child)

    ''' REGULAR sample '''
    def _sampling(self, cur_depth=0):
        selected_modules = []
        if self.depth == self.tot_layers - 1:
            cur_selected_module = self.arch_dist.sample_value().detach().cpu().item()
            selected_modules.append(cur_selected_module)
        else:
            # selected module
            cur_selected_module = self.arch_dist.sample_value().detach().cpu().item()
            selected_modules.append(cur_selected_module)

            chd_selected_modules = self.children[cur_selected_module]._sampling(cur_depth=cur_depth + 1)
            selected_modules += chd_selected_modules
        return selected_modules

    ''' REGULAR sample '''
    def sampling(self, cur_depth=0):
        for jj in range(5):
            cur_sample = self._sampling(cur_depth=cur_depth)
            if not (cur_sample[0] == 3 and cur_sample[1] == 3 and cur_sample[2] == 3):
                return cur_sample
        return [0,0,0]

    def calcu_possi(self, arch_list, cur_depth=0):

        if len(arch_list) == 1:
            cur_possi = float(self.arch_dist.params[0,arch_list[0]].item())
            return cur_possi
        else:
            cur_possi = float(self.arch_dist.params[0,arch_list[0]].item())
            chd_possi = self.children[arch_list[0]].calcu_possi(arch_list[1:], cur_depth + 1)
            return cur_possi * chd_possi

        # return self._sampling(cur_depth=cur_depth)

    ''' SAMPLE baseline architecture combination '''
    def _baseline_sampling(self, cur_depth=0):
        selected_modules = []
        # If self is the leaf node
        if self.depth == self.tot_layers - 1:
            # GET baseline sampling value from the architecture distribution
            cur_selected_module = self.arch_dist.get_baseline_values().detach().cpu().item()
            selected_modules.append(cur_selected_module)
        else:
            # GET architecture baseline value for the current layer
            cur_selected_module = self.arch_dist.get_baseline_values().detach().cpu().item()
            selected_modules.append(cur_selected_module)

            # GET architecture baseline values for children layers
            chd_selected_modules = self.children[cur_selected_module]._baseline_sampling(cur_depth=cur_depth + 1)
            selected_modules += chd_selected_modules
        return selected_modules

    ''' SAMPLE baseline architecture combination '''
    def baseline_sampling(self, cur_depth=0):
        return self._baseline_sampling(cur_depth=cur_depth)

    ''' ACCUMULATE sampling loss by sampled operation dictionary '''
    def update_sampling_dists_by_sampled_dict(self, oper_dist, reward, baseline, lr, accum=True):
        arch_oper_tsr = torch.tensor([oper_dist[0]], dtype=torch.long) # .cuda()

        self.arch_dist.accumulate_sampling_losses_one_sample(value=arch_oper_tsr, reward=reward, baseline=baseline)
        if self.depth < self.tot_layers - 1:
            self.children[oper_dist[0]].update_sampling_dists_by_sampled_dict(oper_dist[1:], reward, baseline, lr, accum=True)
        return

    ''' UPDATE parameters by accumulated loss '''
    def update_by_accum_loss(self, lr, n_samples):
        self.arch_dist.update_params_by_accum_loss(lr=lr, n_samples=n_samples)
        if self.depth < self.tot_layers - 1:
            for chd in self.children:
                chd.update_by_accum_loss(lr=lr, n_samples=n_samples)
        return

    ''' UPDATE sampling distributions by sampled architecture combinations '''
    def update_sampling_dists_by_sampled_dicts(self, oper_dist_list, rewards, baseline, lr):
        n_samples = len(oper_dist_list)
        ''' ACCUMULATE operator selection & transformation selection distributions '''
        for i in range(oper_dist_list.size(0)):
            # for oper_dist in cur_oper_dist_list:
            oper_dist = oper_dist_list[i].tolist()
            cur_reward = rewards[i]
            self.update_sampling_dists_by_sampled_dict(oper_dist, cur_reward, baseline, lr, accum=True)

        ''' UPDATE operator selection & transformation selection distributions by accumulated losses & number of samples '''
        self.update_by_accum_loss(lr=lr, n_samples=n_samples)

    def _collect_params(self,):
        rt_params = {"param_dist": self.arch_dist.params.detach().cpu().numpy()}
        if self.depth < self.tot_layers - 1:
            for i in range(len(self.children)):
                chd_desc = f"chd_{i}"
                rt_params[chd_desc] = self.children[i]._collect_params()
        return rt_params

    ''' COLLECT distribution parameters and save to the file path '''

    def collect_params_and_save(self, file_name):
        tot_params = self._collect_params()
        np.save(file_name, tot_params)


''' FOR loss model version 1 --- distribution tree '''
class DistributionTreeNode:
    def __init__(self, cur_depth=0, nn_grp_opers=4, nn_binary_opers=6, nn_unary_opers=6, nn_in_feat=2, device=None, args=None, forbid_dict=None, preference_dict=None):
        # Distribution tree node construction
        ''' SET parameters '''
        self.nn_grp_opers = nn_grp_opers
        self.nn_binary_opers = nn_binary_opers
        self.nn_unary_opers = nn_unary_opers
        self.nn_in_feat = nn_in_feat
        self.device = device
        self.depth = cur_depth
        self.sampling_tolerance = args.sampling_tolerance
        self.nn_inter_loss = args.nn_inter_loss
        self.forbid_dict = forbid_dict
        self.preference_dict = preference_dict
        ''' SET parameters '''


        if cur_depth == 2:
            # leaf node, only one feature + a unary operation
            self.unary_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_unary_opers, device=device
            ).cuda()
            self.feat_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_in_feat, device=device
            ).cuda()
        elif cur_depth <= 1:
            self.unary_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_unary_opers, device=device
            ).cuda()
            self.grp_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_grp_opers, # + 1 if cur_depth == 1 else nn_grp_opers,
                device=device
            ).cuda()
            if cur_depth == 1:
                # Not a leaf node, not a root node, thus the node's children must be leaf nodes
                self.operator_sampling_dist = MultiMultinomialDistribution(
                    n_dists=1, n_params=1 + nn_binary_opers, device=device
                ).cuda()
                self.unary_children = DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args) #.cuda()
                self.binary_children_right_1, self.binary_children_left_1 = [], []
                for j in range(nn_binary_opers):
                    self.binary_children_left_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                    self.binary_children_right_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
            else:
                self.operator_sampling_dist = MultiMultinomialDistribution(
                    n_dists=1, n_params=1 + 3 * nn_binary_opers, device=device
                ).cuda()

                self.unary_children = DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args)
                self.binary_children_left_1, self.binary_children_right_1 = [], []
                self.binary_children_left_2, self.binary_children_right_2 = [], []
                self.binary_children_left_3, self.binary_children_right_3 = [], []
                for j in range(nn_binary_opers):
                    self.binary_children_left_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                    self.binary_children_right_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                for j in range(nn_binary_opers):
                    self.binary_children_left_2.append(
                        DistributionTreeNode(1, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                             nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                    self.binary_children_right_2.append(
                        DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                             nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                for j in range(nn_binary_opers):
                    self.binary_children_left_3.append(
                        DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                             nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                    self.binary_children_right_3.append(
                        DistributionTreeNode(1, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                             nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))

    ''' FOR baseline sampling '''
    def _get_topk_indexes(self, tsr, k):
        topk_possi, topk_idx = torch.topk(tsr, k=min(k, tsr.size(-1)), largest=True)
        # print("=== Topk sampling resulted possi's size & idx's size ===")
        topk_possi = topk_possi.squeeze(0)
        topk_idx = topk_idx.squeeze(0)
        # print(topk_possi.size(), topk_idx.size())
        return topk_possi.tolist(), topk_idx.tolist()

    ''' FOR baseline sampling '''
    def _sampling_via_topk_dists_combinations(self, topk=2, cur_depth=0):
        if cur_depth == 2:
            # SAMPLE topk possibility
            cur_topk_unaries_possi, cur_topk_unaries_idx = self._get_topk_indexes(self.unary_sampling_dist.params, k=topk)
            cur_topk_feat_possi, cur_topk_feat_idx = self._get_topk_indexes(self.feat_sampling_dist.params, k=topk)
            rt_dict_list = []
            for i_uop, (cur_uop_possi, cur_uop) in enumerate(zip(cur_topk_unaries_possi, cur_topk_unaries_idx)):
                for i_feat, (cur_feat_possi, cur_feat) in enumerate(zip(cur_topk_feat_possi, cur_topk_feat_idx)):
                    cur_rt_dict = {"uop": cur_uop, "oper": cur_feat}
                    rt_dict_list.append((cur_rt_dict, cur_uop_possi * cur_feat_possi))
        else:
            rt_dict_list = []
            cur_topk_gop_possi, cur_topk_gop_idx = self._get_topk_indexes(self.grp_sampling_dist.params, k=topk)
            cur_topk_uop_possi, cur_topk_uop_idx = self._get_topk_indexes(self.unary_sampling_dist.params, k=topk)
            cur_topk_oper_possi, cur_topk_oper_idx = self._get_topk_indexes(self.operator_sampling_dist.params, k=topk)
            for i_oper, (cur_oper_possi, cur_oper) in enumerate(zip(cur_topk_oper_possi, cur_topk_oper_idx)):
                if cur_oper == 0:
                    chd_rt_dict_list = self.unary_children._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2)
                    for i_gop, (cur_gop_possi, cur_gop) in enumerate(zip(cur_topk_gop_possi, cur_topk_gop_idx)):
                        for i_uop, (cur_uop_possi, cur_uop) in enumerate(zip(cur_topk_uop_possi, cur_topk_uop_idx)):
                            for i_chd_dict, chd_dict in enumerate(chd_rt_dict_list):
                                cur_rt_dict = {"gop": cur_gop, "uop": cur_uop, "chd": chd_dict[0]}
                                cur_rt_dict_possi = chd_dict[1] * cur_gop_possi * cur_uop_possi * cur_oper_possi
                                rt_dict_list.append((cur_rt_dict, cur_rt_dict_possi))
                else:
                    if cur_oper <= self.nn_binary_opers:
                        lft_chd, rgt_chd = self.binary_children_left_1[cur_oper - 1], self.binary_children_right_1[cur_oper - 1]
                        lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2), rgt_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2)
                    elif cur_oper <= 2 * self.nn_binary_opers:
                        lft_chd, rgt_chd = self.binary_children_left_2[
                                               cur_oper - self.nn_binary_opers - 1], \
                                           self.binary_children_right_2[
                                               cur_oper - self.nn_binary_opers - 1]
                        lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=1), rgt_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2)
                    elif cur_oper <= 3 * self.nn_binary_opers:
                        lft_chd, rgt_chd = self.binary_children_left_3[cur_oper - 2 * self.nn_binary_opers - 1], self.binary_children_right_3[cur_oper - 2 * self.nn_binary_opers - 1]
                        lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2), rgt_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=1)
                    else:
                        raise ValueError(f"Unrecognized sampled oper_idx: {cur_oper}.")

                    for i_gop, (cur_gop_possi, cur_gop) in enumerate(zip(cur_topk_gop_possi, cur_topk_gop_idx)):
                        for i_uop, (cur_uop_possi, cur_uop) in enumerate(zip(cur_topk_uop_possi, cur_topk_uop_idx)):
                            for i_lft_chd, lft_chd_dict in enumerate(lft_chd_dict_list):
                                for i_rgt_chd, rgt_chd_dict in enumerate(rgt_chd_dict_list):
                                    cur_rt_dict = {"gop": cur_gop, "uop": cur_uop, "bop": cur_oper, "lft_chd": lft_chd_dict[0], "rgt_chd": rgt_chd_dict[0]}
                                    cur_rt_dict_possi = lft_chd_dict[1] * rgt_chd_dict[1] * cur_gop_possi * cur_uop_possi * cur_oper_possi
                                    rt_dict_list.append((cur_rt_dict, cur_rt_dict_possi))

        # CLAMP candidate operation combination dictionariy list to `sampled_oper_k * 5`, since the maximum tolerance towards repeating sampling is 5.
        rt_dict_list = sorted(rt_dict_list, key=lambda i: i[1], reverse=True)
        rt_dict_list = rt_dict_list[: min(len(rt_dict_list), self.nn_inter_loss * self.sampling_tolerance)]
        return rt_dict_list

    ''' FOR baseline sampling '''
    def _get_ranked_operations_and_ratios(self, topk=2, sampled_oper_k=4):
        sampled_oper_k = self.nn_inter_loss
        oper_dict_possi_list = self._sampling_via_topk_dists_combinations(topk=topk, cur_depth=0)
        # NEED NOT to sort the possibility list again since the returned list has already be solved
        oper_dict_possi_list = sorted(oper_dict_possi_list, key=lambda i: i[1], reverse=True)
        res_oper_dict_list = []

        # REMOVE those non-invalid features
        for i, oper_dict_possi in enumerate(oper_dict_possi_list):
            if loss_model.get_pred_feat_dim_from_oper_dict(oper_dict_possi[0]) is not None:
                res_oper_dict_list.append(oper_dict_possi[0])
                if len(res_oper_dict_list) == sampled_oper_k:
                    break
        return res_oper_dict_list

    def _sampling(self, cur_depth=0):
        if cur_depth == 2:
            cur_unary_sampling = self.unary_sampling_dist.sample_value().detach().item()
            cur_feat = self.feat_sampling_dist.sample_value().detach().item()
            rt_dict = {"uop": cur_unary_sampling, "oper": cur_feat}
        else:
            cur_grp_sampling = self.grp_sampling_dist.sample_value().detach().item()
            cur_unary_sampling = self.unary_sampling_dist.sample_value().detach().item()
            cur_operator_sampling = self.operator_sampling_dist.sample_value().detach().item()
            if cur_operator_sampling == 0:
                chd_dict = self.unary_children._sampling(cur_depth=2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "chd": chd_dict}
            elif cur_operator_sampling <= self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_1[cur_operator_sampling - 1], self.binary_children_right_1[cur_operator_sampling - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(2), rgt_chd._sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling, "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 2 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_2[cur_operator_sampling - self.nn_binary_opers - 1], self.binary_children_right_2[
                    cur_operator_sampling - self.nn_binary_opers - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(1), rgt_chd._sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 3 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_3[cur_operator_sampling - 2 * self.nn_binary_opers - 1], \
                                   self.binary_children_right_3[
                                       cur_operator_sampling - 2 * self.nn_binary_opers - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(2), rgt_chd._sampling(1)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            else:
                raise ValueError(f"Invalid sampled operator index: {cur_operator_sampling}.")
        return rt_dict

    def sampling(self, cur_depth=0, nn_max=5):
        nn_max = self.sampling_tolerance
        for i in range(nn_max):
            rt_dict = self._sampling(cur_depth=0)
            if loss_model.get_pred_feat_dim_from_oper_dict(rt_dict, in_feat_dim=self.nn_in_feat, nn_binary_opers=self.nn_binary_opers) is not None:
                return rt_dict
        return rt_dict

    ''' SAMPLE baseline operation dictionary '''
    def sample_basline_oper_dict_list(self):
        baseline_oper_dict_list = self._get_ranked_operations_and_ratios(topk=2)
        return baseline_oper_dict_list

    ''' UPDATE parameters by sampled dictionaries '''
    def update_sampling_dists_by_sampled_dict(self, oper_dist, reward, baseline, lr, accum=True):
        # gop = oper_dist["gop"]
        # cur_gop_tsr = torch.tensor([gop], dtype=torch.long).cuda()
        # self.grp_sampling_dist.accumulate_sampling_losses_one_sample(
        #     cur_gop_tsr, reward, baseline
        # )

        if "oper" in oper_dist:
            # leaf node
            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long).cuda()
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            oper = oper_dist["oper"]
            cur_oper_tsr = torch.tensor([oper], dtype=torch.long).cuda()
            self.feat_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )
        elif "chd" in oper_dist:
            chd_oper_dict = oper_dist["chd"]

            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long).cuda()
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            gop = oper_dist["gop"]
            cur_gop_tsr = torch.tensor([gop], dtype=torch.long).cuda()
            self.grp_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_gop_tsr, reward, baseline
            )

            cur_oper_tsr = torch.tensor([0], dtype=torch.long).cuda()
            self.operator_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )

            self.unary_children.update_sampling_dists_by_sampled_dict(chd_oper_dict, reward, baseline, lr)
        else:
            lft_chd_oper_dict, rgt_chd_oper_dict = oper_dist["lft_chd"], oper_dist["rgt_chd"]

            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long).cuda()
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            gop = oper_dist["gop"]
            cur_gop_tsr = torch.tensor([gop], dtype=torch.long).cuda()
            self.grp_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_gop_tsr, reward, baseline
            )

            bop = oper_dist["bop"]
            cur_oper_tsr = torch.tensor([bop], dtype=torch.long).cuda()
            self.operator_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )

            if bop <= self.nn_binary_opers:
                self.binary_children_left_1[bop - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_1[bop - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)
            elif bop <= self.nn_binary_opers * 2:
                self.binary_children_left_2[(bop - self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_2[(bop - self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)
            elif bop <= self.nn_binary_opers * 3:
                self.binary_children_left_3[(bop - 2 * self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_3[(bop - 2 * self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)

    def update_by_accum_loss(self, lr):
        # self.grp_sampling_dist.update_params_by_accum_loss(lr=lr)
        if self.depth == 2:
            self.feat_sampling_dist.update_params_by_accum_loss(lr=lr)
            self.unary_sampling_dist.update_params_by_accum_loss(lr=lr)
        else:
            self.grp_sampling_dist.update_params_by_accum_loss(lr=lr)
            self.operator_sampling_dist.update_params_by_accum_loss(lr=lr)
            self.unary_sampling_dist.update_params_by_accum_loss(lr=lr)
            self.unary_children.update_by_accum_loss(lr=lr)
            if self.depth == 1:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1, self.binary_children_right_1)):
                    b_lft_chd.update_by_accum_loss(lr=lr)
                    b_rgt_chd.update_by_accum_loss(lr=lr)
            else:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1 + self.binary_children_left_2 + self.binary_children_left_3, self.binary_children_right_1 + self.binary_children_right_2 + self.binary_children_right_3)):
                    b_lft_chd.update_by_accum_loss(lr=lr)
                    b_rgt_chd.update_by_accum_loss(lr=lr)

    ''' THIS function should only be called for the root node '''
    def update_sampling_dists_by_sampled_dicts(self, oper_dist_list, rewards, baseline, lr):
        n_samples = len(oper_dist_list)
        ''' ACCUMULATE operator selection & transformation selection distributions '''
        for i, cur_oper_dist_list in enumerate(oper_dist_list):
            for oper_dist in cur_oper_dist_list:
                cur_reward = rewards[i]
                self.update_sampling_dists_by_sampled_dict(oper_dist, cur_reward, baseline, lr, accum=True)

        ''' UPDATE operator selection & transformation selection distributions by accumulated losses & number of samples '''
        self.update_by_accum_loss(lr=lr)

    ''' PRINT parameters to screen '''
    def print_params(self):
        if self.depth == 2:
            # print("grp_sampling_dist.params = ")
            # print(self.grp_sampling_dist.params)
            print("unary_sampling_dist.params = ")
            print(self.unary_sampling_dist.params)
            print("feat_sampling_dist.params = ")
            print(self.feat_sampling_dist.params)
        else:
            print("grp_sampling_dist.params = ")
            print(self.grp_sampling_dist.params)
            print("unary_sampling_dist.params = ")
            print(self.unary_sampling_dist.params)
            print("operator_sampling_dist.params = ")
            print(self.operator_sampling_dist.params)
            print(f"unary child's params")
            self.unary_children.print_params()
            if self.depth == 1:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1, self.binary_children_right_1)):
                    print(f"{i}-th left binary child's params")
                    b_lft_chd.print_params()
                    print(f"{i}-th right binary child's params")
                    b_rgt_chd.print_params()
            else:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1 + self.binary_children_left_2 + self.binary_children_left_3, self.binary_children_right_1 + self.binary_children_right_2 + self.binary_children_right_3)):
                    print(f"{i}-th left binary child's params")
                    b_lft_chd.print_params()
                    print(f"{i}-th right binary child's params")
                    b_rgt_chd.print_params()

    ''' PRINT parameters to file '''
    def print_params_to_file(self, wf):
        if self.depth == 2:
            # wf.write("grp_sampling_dist.params" + "\n")
            # wf.write(f"{self.grp_sampling_dist.params}")
            wf.write("unary_sampling_dist.params" + "\n")
            wf.write(f"{self.unary_sampling_dist.params}")
            wf.write("feat_sampling_dist.params" + "\n")
            wf.write(f"{self.feat_sampling_dist.params}")
        else:
            wf.write("grp_sampling_dist.params" + "\n")
            wf.write(f"{self.grp_sampling_dist.params}")
            wf.write("unary_sampling_dist.params" + "\n")
            wf.write(f"{self.unary_sampling_dist.params}")
            wf.write("operator_sampling_dist.params" + "\n")
            wf.write(f"{self.operator_sampling_dist.params}")
            wf.write(f"unary child's params" + "\n")
            self.unary_children.print_params_to_file(wf)
            if self.depth == 1:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1, self.binary_children_right_1)):
                    wf.write(f"{i}-th left binary child's params" + "\n")
                    b_lft_chd.print_params_to_file(wf)
                    wf.write(f"{i}-th right binary child's params" + "\n")
                    b_rgt_chd.print_params_to_file(wf)
            else:
                for i, (b_lft_chd, b_rgt_chd) in enumerate(zip(self.binary_children_left_1 + self.binary_children_left_2 + self.binary_children_left_3, self.binary_children_right_1 + self.binary_children_right_2 + self.binary_children_right_3)):
                    wf.write(f"{i}-th left binary child's params" + "\n")
                    b_lft_chd.print_params_to_file(wf)
                    wf.write(f"{i}-th right binary child's params" + "\n")
                    b_rgt_chd.print_params_to_file(wf)


class DistributionTreeNodeV2:
    def __init__(self, cur_depth=0, nn_grp_opers=4, nn_binary_opers=6, nn_unary_opers=6, nn_in_feat=2, device=None, args=None, forbid_dict=None, preference_dict=None, preload_params=None):
        # Distribution tree node construction
        ''' SET parameters '''
        self.nn_grp_opers = nn_grp_opers
        self.nn_binary_opers = nn_binary_opers
        self.nn_unary_opers = nn_unary_opers
        self.nn_in_feat = nn_in_feat
        self.device = device
        self.depth = cur_depth
        self.sampling_tolerance = args.sampling_tolerance
        self.nn_inter_loss = args.nn_inter_loss
        self.forbid_dict = forbid_dict
        self.preference_dict = preference_dict
        ''' SET parameters '''

        if cur_depth == 2:
            # leaf node, only one feature + a unary operation
            # #### Unary operator sampling distribution ####
            self.unary_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_unary_opers, device=device
            ) # .cuda()

            if preload_params is not None:
                self.unary_sampling_dist.load_params(torch.from_numpy(preload_params["uop"]))

            # #### Get feature sampling distribution for each kind of unary operator ####
            self.feat_sampling_dist = []
            for i_uop in range(self.nn_unary_opers):
                self.feat_sampling_dist.append(MultiMultinomialDistribution(
                        n_dists=1, n_params=nn_in_feat, device=device
                    ) # .cuda()
                )
                if preload_params is not None:
                    self.feat_sampling_dist[i_uop].load_params(torch.from_numpy(preload_params["fop"][i_uop]))

        # {"uop": self.unary_sampling_dist.params.detach().cpu().numpy(),
        #  "gop": [cur_gop_dist.params.detach().cpu().numpy() for cur_gop_dist in self.grp_sampling_dist],
        #  "oop": {cur_oper: self.operator_sampling_dist[cur_oper].params.detach().cpu().numpy() for cur_oper in
        #          self.operator_sampling_dist},
        #  "unary_children": {cur_oper: self.unary_children[cur_oper]._collect_params() for cur_oper in
        #                     self.unary_children}}
        elif cur_depth <= 1:
            self.unary_sampling_dist = MultiMultinomialDistribution(
                n_dists=1, n_params=nn_unary_opers, device=device
            )
            if preload_params is not None:
                self.unary_sampling_dist.load_params(torch.from_numpy(preload_params["uop"]))
            self.grp_sampling_dist = []
            for i_uop in range(nn_unary_opers):
                self.grp_sampling_dist.append(MultiMultinomialDistribution(
                    n_dists=1, n_params=nn_grp_opers,
                    device=device
                ).cuda())
                if preload_params is not None:
                    self.grp_sampling_dist[i_uop].load_params(torch.from_numpy(preload_params["gop"][i_uop]))

            if cur_depth == 1:

                self.operator_sampling_dist = {}
                self.unary_children = {}
                self.binary_children_right_1, self.binary_children_left_1 = {}, {}

                for i_uop in range(nn_unary_opers):
                    for i_gop in range(nn_grp_opers):
                        cur_operator_sampling_dist = MultiMultinomialDistribution(
                            n_dists=1, n_params=1 + nn_binary_opers, device=device
                        ) # .cuda()
                        cur_unary_child = DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["unary_children"][(i_uop, i_gop)]) #.cuda()
                        cur_binary_children_right_1, cur_binary_children_left_1 = [], []
                        for j in range(nn_binary_opers):
                            cur_binary_children_left_1.append(DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["lft_children"][(i_uop, i_gop)][j]))
                            cur_binary_children_right_1.append(DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["rgt_children"][(i_uop, i_gop)][j]))
                        self.binary_children_right_1[(i_uop, i_gop)] = cur_binary_children_right_1
                        self.binary_children_left_1[(i_uop, i_gop)] = cur_binary_children_left_1
                        self.operator_sampling_dist[(i_uop, i_gop)] = cur_operator_sampling_dist
                        self.unary_children[(i_uop, i_gop)] = cur_unary_child
                        if preload_params is not None:
                            self.operator_sampling_dist[(i_uop, i_gop)].load_params(torch.from_numpy(preload_params["oop"][(i_uop, i_gop)]))
                # # Not a leaf node, not a root node, thus the node's children must be leaf nodes
                # self.operator_sampling_dist = MultiMultinomialDistribution(
                #     n_dists=1, n_params=1 + nn_binary_opers, device=device
                # ).cuda()
                # self.unary_children = DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args) #.cuda()
                # self.binary_children_right_1, self.binary_children_left_1 = [], []
                # for j in range(nn_binary_opers):
                #     self.binary_children_left_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
                #     self.binary_children_right_1.append(DistributionTreeNode(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers, nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat, device=device, args=args))
            else:
                self.operator_sampling_dist = {}
                self.unary_children = {}
                self.binary_children_right_1, self.binary_children_right_2, self.binary_children_right_3 = {}, {}, {}
                self.binary_children_left_1, self.binary_children_left_2, self.binary_children_left_3 = {}, {}, {}

                for i_uop in range(nn_unary_opers):
                    for i_gop in range(nn_grp_opers):
                        cur_operator_sampling_dist = MultiMultinomialDistribution(
                            n_dists=1, n_params=1 + nn_binary_opers * 3, device=device
                        ) # .cuda()
                        cur_unary_child = DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers,
                                                               nn_binary_opers=nn_binary_opers,
                                                               nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                               device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["unary_children"][(i_uop, i_gop)])  # .cuda()
                        cur_binary_children_right_1, cur_binary_children_left_1 = [], []
                        cur_binary_children_right_2, cur_binary_children_left_2 = [], []
                        cur_binary_children_right_3, cur_binary_children_left_3 = [], []

                        for j in range(nn_binary_opers):
                            cur_binary_children_left_1.append(
                                DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["lft_children_1"][(i_uop, i_gop)][j]))
                            cur_binary_children_right_1.append(
                                DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["rgt_children_1"][(i_uop, i_gop)][j]))
                            cur_binary_children_left_2.append(
                                DistributionTreeNodeV2(1, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["lft_children_2"][(i_uop, i_gop)][j]))
                            cur_binary_children_right_2.append(
                                DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["rgt_children_2"][(i_uop, i_gop)][j]))
                            cur_binary_children_left_3.append(
                                DistributionTreeNodeV2(2, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["lft_children_3"][(i_uop, i_gop)][j]))
                            cur_binary_children_right_3.append(
                                DistributionTreeNodeV2(1, nn_grp_opers=nn_grp_opers, nn_binary_opers=nn_binary_opers,
                                                     nn_unary_opers=nn_unary_opers, nn_in_feat=nn_in_feat,
                                                     device=device, args=args, forbid_dict=self.forbid_dict, preference_dict=self.preference_dict, preload_params=None if preload_params is None else preload_params["rgt_children_3"][(i_uop, i_gop)][j]))
                        self.binary_children_right_1[(i_uop, i_gop)] = cur_binary_children_right_1
                        self.binary_children_right_2[(i_uop, i_gop)] = cur_binary_children_right_2
                        self.binary_children_right_3[(i_uop, i_gop)] = cur_binary_children_right_3
                        self.binary_children_left_1[(i_uop, i_gop)] = cur_binary_children_left_1
                        self.binary_children_left_2[(i_uop, i_gop)] = cur_binary_children_left_2
                        self.binary_children_left_3[(i_uop, i_gop)] = cur_binary_children_left_3

                        self.operator_sampling_dist[(i_uop, i_gop)] = cur_operator_sampling_dist
                        self.unary_children[(i_uop, i_gop)] = cur_unary_child

                        if preload_params is not None:
                            self.operator_sampling_dist[(i_uop, i_gop)].load_params(torch.from_numpy(preload_params["oop"][(i_uop, i_gop)]))

    ''' FOR baseline sampling '''
    def _get_topk_indexes(self, tsr, k):
        topk_possi, topk_idx = torch.topk(tsr, k=min(k, tsr.size(-1)), largest=True)
        topk_possi = topk_possi.squeeze(0)
        topk_idx = topk_idx.squeeze(0)
        # print(topk_possi.size(), topk_idx.size())
        return topk_possi.tolist(), topk_idx.tolist()

    ''' FOR baseline sampling '''
    def _sampling_via_topk_dists_combinations(self, topk=2, cur_depth=0):
        if cur_depth == 2:
            cur_topk_unaries_possi, cur_topk_unaries_idx = self._get_topk_indexes(self.unary_sampling_dist.params, k=topk)
            rt_dict_list = []
            for i_uop, (cur_uop_possi, cur_uop) in enumerate(zip(cur_topk_unaries_possi, cur_topk_unaries_idx)):
                cur_topk_feat_possi, cur_topk_feat_idx = self._get_topk_indexes(self.feat_sampling_dist[i_uop].params, k=topk)
                for i_feat, (cur_feat_possi, cur_feat) in enumerate(zip(cur_topk_feat_possi, cur_topk_feat_idx)):
                    cur_rt_dict = {"uop": cur_uop, "oper": cur_feat}
                    rt_dict_list.append((cur_rt_dict, cur_uop_possi * cur_feat_possi))
        else:
            rt_dict_list = []
            cur_topk_uop_possi, cur_topk_uop_idx = self._get_topk_indexes(self.unary_sampling_dist.params, k=topk)

            for i_uop, (cur_uop_possi, cur_uop) in enumerate(zip(cur_topk_uop_possi, cur_topk_uop_idx)):
                cur_topk_gop_possi, cur_topk_gop_idx = self._get_topk_indexes(self.grp_sampling_dist[i_uop].params, k=topk)
                for i_gop, (cur_gop_possi, cur_gop) in enumerate(zip(cur_topk_gop_possi, cur_topk_gop_idx)):
                    cur_topk_oper_possi, cur_topk_oper_idx = self._get_topk_indexes(self.operator_sampling_dist[(i_uop, i_gop)].params, k=topk)
                    for i_oper, (cur_oper_possi, cur_oper) in enumerate(zip(cur_topk_oper_possi, cur_topk_oper_idx)):
                        if cur_oper == 0:
                            chd_rt_dict_list = self.unary_children[(i_uop, i_gop)]._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2)
                            for i_chd_dict, chd_dict in enumerate(chd_rt_dict_list):
                                cur_rt_dict = {"gop": cur_gop, "uop": cur_uop, "chd": chd_dict[0]}
                                cur_rt_dict_possi = chd_dict[1] * cur_gop_possi * cur_uop_possi * cur_oper_possi
                                rt_dict_list.append((cur_rt_dict, cur_rt_dict_possi))
                        elif cur_oper <= self.nn_binary_opers:
                            lft_chd, rgt_chd = self.binary_children_left_1[(i_uop, i_gop)][cur_oper - 1], self.binary_children_right_1[(i_uop, i_gop)][cur_oper - 1]
                            lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2), rgt_chd._sampling_via_topk_dists_combinations(topk=topk, cur_depth=2)
                        elif cur_oper <= 2 * self.nn_binary_opers:
                            lft_chd, rgt_chd = self.binary_children_left_2[(i_uop, i_gop)][cur_oper - self.nn_binary_opers - 1], self.binary_children_right_2[(i_uop, i_gop)][cur_oper - self.nn_binary_opers - 1]
                            lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(
                                topk=topk, cur_depth=1), rgt_chd._sampling_via_topk_dists_combinations(topk=topk,
                                                                                                       cur_depth=2)
                        elif cur_oper <= 3 * self.nn_binary_opers:
                            lft_chd, rgt_chd = self.binary_children_left_3[(i_uop, i_gop)][cur_oper - 2 * self.nn_binary_opers - 1], self.binary_children_right_3[(i_uop, i_gop)][cur_oper - 2 * self.nn_binary_opers - 1]
                            lft_chd_dict_list, rgt_chd_dict_list = lft_chd._sampling_via_topk_dists_combinations(
                                topk=topk, cur_depth=2), rgt_chd._sampling_via_topk_dists_combinations(topk=topk,
                                                                                                       cur_depth=1)
                        else:
                            raise ValueError(f"Unrecognized sampled oper_idx: {cur_oper}.")
                        if cur_oper > 0:
                            for i_lft_chd, lft_chd_dict in enumerate(lft_chd_dict_list):
                                for i_rgt_chd, rgt_chd_dict in enumerate(rgt_chd_dict_list):
                                    cur_rt_dict = {"gop": cur_gop, "uop": cur_uop, "bop": cur_oper, "lft_chd": lft_chd_dict[0], "rgt_chd": rgt_chd_dict[0]}
                                    cur_rt_dict_possi = lft_chd_dict[1] * rgt_chd_dict[1] * cur_gop_possi * cur_uop_possi * cur_oper_possi
                                    rt_dict_list.append((cur_rt_dict, cur_rt_dict_possi))

        # CLAMP candidate operation combination dictionariy list to `sampled_oper_k * 5`, since the maximum tolerance towards repeating sampling is 5.
        rt_dict_list = sorted(rt_dict_list, key=lambda i: i[1], reverse=True)
        rt_dict_list = rt_dict_list[: min(len(rt_dict_list), self.nn_inter_loss * self.sampling_tolerance)]
        return rt_dict_list

    ''' FOR baseline sampling '''
    def _get_ranked_operations_and_ratios(self, topk=2, sampled_oper_k=4):
        sampled_oper_k = self.nn_inter_loss
        oper_dict_possi_list = self._sampling_via_topk_dists_combinations(topk=topk, cur_depth=self.depth)
        oper_dict_possi_list = sorted(oper_dict_possi_list, key=lambda i: i[1], reverse=True)
        res_oper_dict_list = []

        for i, oper_dict_possi in enumerate(oper_dict_possi_list):
            if loss_model.get_pred_feat_dim_from_oper_dict(oper_dict_possi[0]) is not None:
                res_oper_dict_list.append(oper_dict_possi[0])
                if len(res_oper_dict_list) == sampled_oper_k:
                    break
        return res_oper_dict_list

    ''' INNER sampling function '''
    def _sampling(self, cur_depth=0):
        if cur_depth == 2:
            cur_unary_sampling = self.unary_sampling_dist.sample_value().detach().item()
            cur_feat = self.feat_sampling_dist[cur_unary_sampling].sample_value().detach().item()
            rt_dict = {"uop": cur_unary_sampling, "oper": cur_feat}
        else:
            cur_unary_sampling = self.unary_sampling_dist.sample_value().detach().item()
            cur_grp_sampling = self.grp_sampling_dist[cur_unary_sampling].sample_value().detach().item()

            cur_operator_sampling = self.operator_sampling_dist[(cur_unary_sampling, cur_grp_sampling)].sample_value().detach().item()
            if cur_operator_sampling == 0:
                chd_dict = self.unary_children[(cur_unary_sampling, cur_grp_sampling)]._sampling(cur_depth=2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "chd": chd_dict}
            elif cur_operator_sampling <= self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_1[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - 1], self.binary_children_right_1[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(2), rgt_chd._sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling, "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 2 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_2[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - self.nn_binary_opers - 1], self.binary_children_right_2[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - self.nn_binary_opers - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(1), rgt_chd._sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 3 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_3[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - 2 * self.nn_binary_opers - 1], self.binary_children_right_3[(cur_unary_sampling, cur_grp_sampling)][cur_operator_sampling - 2 * self.nn_binary_opers - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd._sampling(2), rgt_chd._sampling(1)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            else:
                raise ValueError(f"Invalid sampled operator index: {cur_operator_sampling}.")
        return rt_dict

    ''' ROOT sampling function (should only be called on the root node) '''
    def sampling(self, cur_depth=0, nn_max=5):
        nn_max = self.sampling_tolerance
        for i in range(nn_max):
            rt_dict = self._sampling(cur_depth=self.depth)
            if loss_model.get_pred_feat_dim_from_oper_dict(rt_dict, in_feat_dim=self.nn_in_feat, nn_binary_opers=self.nn_binary_opers) is not None:
                return rt_dict
        return None
        # return rt_dict

    ''' Not used currently '''
    def baseline_sampling(self, cur_depth=0):
        if cur_depth == 2:
            cur_unary_sampling = self.unary_sampling_dist.get_baseline_values().detach().item()
            cur_feat = self.feat_sampling_dist.get_baseline_values().detach().item()
            rt_dict = {"uop": cur_unary_sampling, "oper": cur_feat}
        else:
            cur_grp_sampling = self.grp_sampling_dist.get_baseline_values().detach().item()
            cur_unary_sampling = self.unary_sampling_dist.get_baseline_values().detach().item()
            cur_operator_sampling = self.operator_sampling_dist.get_baseline_values().detach().item()
            if cur_operator_sampling == 0:
                chd_dict = self.unary_children.baseline_sampling(cur_depth=2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "chd": chd_dict}
            elif cur_operator_sampling <= self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_1[cur_operator_sampling - 1], self.binary_children_right_1[
                    cur_operator_sampling - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd.baseline_sampling(2), rgt_chd.baseline_sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 2 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_2[(cur_operator_sampling - self.nn_binary_opers) - 1], \
                                   self.binary_children_right_2[
                                       (cur_operator_sampling - self.nn_binary_opers) - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd.baseline_sampling(1), rgt_chd.baseline_sampling(2)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            elif cur_operator_sampling <= 3 * self.nn_binary_opers:
                lft_chd, rgt_chd = self.binary_children_left_3[(cur_operator_sampling - 2 * self.nn_binary_opers) - 1], \
                                   self.binary_children_right_3[
                                       (cur_operator_sampling - 2 * self.nn_binary_opers) - 1]
                lft_chd_dict, rgt_chd_dict = lft_chd.baseline_sampling(2), rgt_chd.baseline_sampling(1)
                rt_dict = {"gop": cur_grp_sampling, "uop": cur_unary_sampling, "bop": cur_operator_sampling,
                           "lft_chd": lft_chd_dict, "rgt_chd": rgt_chd_dict}
            else:
                raise ValueError(f"Invalid sampled operator index: {cur_operator_sampling}.")
        return rt_dict

    ''' SAMPLE baseline operation dictionaries '''
    def sample_basline_oper_dict_list(self):
        baseline_oper_dict_list = self._get_ranked_operations_and_ratios(topk=2)
        return baseline_oper_dict_list

    ''' ACCUMULATE sampling loss from the input sample (sampled operation dict, corresponding reward and baseline) '''
    def update_sampling_dists_by_sampled_dict(self, oper_dist, reward, baseline, lr, accum=True):
        # gop = oper_dist["gop"]
        # cur_gop_tsr = torch.tensor([gop], dtype=torch.long).cuda()
        # self.grp_sampling_dist.accumulate_sampling_losses_one_sample(
        #     cur_gop_tsr, reward, baseline
        # )

        if "oper" in oper_dist:
            # leaf node
            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long) # .cuda()
            # How to update distributional parameters when there are multiple samples?
            # ACCUMULATE sampling loss from each sample
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            oper = oper_dist["oper"]
            cur_oper_tsr = torch.tensor([oper], dtype=torch.long) # .cuda()
            # ACCUMULATE sampling loss from each sample
            self.feat_sampling_dist[uop].accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )
        elif "chd" in oper_dist:
            chd_oper_dict = oper_dist["chd"]

            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long) # .cuda()
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            gop = oper_dist["gop"]
            cur_gop_tsr = torch.tensor([gop], dtype=torch.long) # .cuda()
            self.grp_sampling_dist[uop].accumulate_sampling_losses_one_sample(
                cur_gop_tsr, reward, baseline
            )

            cur_oper_tsr = torch.tensor([0], dtype=torch.long) # .cuda()
            self.operator_sampling_dist[(uop, gop)].accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )

            self.unary_children[(uop, gop)].update_sampling_dists_by_sampled_dict(chd_oper_dict, reward, baseline, lr)
        else:
            lft_chd_oper_dict, rgt_chd_oper_dict = oper_dist["lft_chd"], oper_dist["rgt_chd"]

            uop = oper_dist["uop"]
            cur_uop_tsr = torch.tensor([uop], dtype=torch.long) # .cuda()
            self.unary_sampling_dist.accumulate_sampling_losses_one_sample(
                cur_uop_tsr, reward, baseline
            )

            gop = oper_dist["gop"]
            cur_gop_tsr = torch.tensor([gop], dtype=torch.long) # .cuda()
            self.grp_sampling_dist[uop].accumulate_sampling_losses_one_sample(
                cur_gop_tsr, reward, baseline
            )

            bop = oper_dist["bop"]
            cur_oper_tsr = torch.tensor([bop], dtype=torch.long) # .cuda()
            self.operator_sampling_dist[(uop, gop)].accumulate_sampling_losses_one_sample(
                cur_oper_tsr, reward, baseline
            )

            if bop <= self.nn_binary_opers:
                self.binary_children_left_1[(uop, gop)][bop - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_1[(uop, gop)][bop - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)
            elif bop <= self.nn_binary_opers * 2:
                self.binary_children_left_2[(uop, gop)][(bop - self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_2[(uop, gop)][(bop - self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)
            elif bop <= self.nn_binary_opers * 3:
                self.binary_children_left_3[(uop, gop)][(bop - 2 * self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(lft_chd_oper_dict, reward, baseline, lr)
                self.binary_children_right_3[(uop, gop)][(bop - 2 * self.nn_binary_opers) - 1].update_sampling_dists_by_sampled_dict(rgt_chd_oper_dict, reward, baseline, lr)

    ''' UPDATE parameters by accumulated losses layer by layer '''
    def update_by_accum_loss(self, lr, n_samples):
        if self.depth == 2:
            self.unary_sampling_dist.update_params_by_accum_loss(lr=lr, n_samples=n_samples)
            for i, cur_feat_sampling_dist in enumerate(self.feat_sampling_dist):
                # if ("uop:{}")
                if self.forbid_dict is not None and f"uop_{i}:feat" in self.forbid_dict:
                    cur_forbid_dict = self.forbid_dict[f"uop_{i}:feat"]
                else:
                    cur_forbid_dict = None
                if self.preference_dict is not None and f"uop_{i}:feat" in self.preference_dict:
                    cur_preference_dict = self.preference_dict[f"uop_{i}:feat"]
                else:
                    cur_preference_dict = None
                cur_feat_sampling_dist.update_params_by_accum_loss(lr=lr, n_samples=n_samples, forbid_dict=cur_forbid_dict, preference_dict=cur_preference_dict)
        else:
            ''' UPDATE sampling distribution by accumulated loss '''
            self.unary_sampling_dist.update_params_by_accum_loss(lr=lr, n_samples=n_samples)
            for i, cur_grp_sampling_dist in enumerate(self.grp_sampling_dist):
                cur_forbid_dict, cur_preference_dict = None, None
                if self.forbid_dict is not None and f"uop_{i}:grp" in self.forbid_dict:
                    cur_forbid_dict = self.forbid_dict[f"uop_{i}:grp"]
                if self.preference_dict is not None and f"uop_{i}:grp" in self.preference_dict:
                    cur_preference_dict = self.preference_dict[f"uop_{i}:grp"]
                cur_grp_sampling_dist.update_params_by_accum_loss(lr=lr, n_samples=n_samples, forbid_dict=cur_forbid_dict, preference_dict=cur_preference_dict)
            for kk in self.operator_sampling_dist:
                cur_gop = kk[1]
                cur_gene_str = f"gop_{cur_gop}:u_b_op"
                cur_forbid_dict, cur_preference_dict = None, None
                if self.forbid_dict is not None and cur_gene_str in self.forbid_dict:
                    cur_forbid_dict = self.forbid_dict[cur_gene_str]
                if self.preference_dict is not None and cur_gene_str in self.preference_dict:
                    cur_preference_dict = self.preference_dict[cur_gene_str]
                self.operator_sampling_dist[kk].update_params_by_accum_loss(lr=lr, n_samples=n_samples, forbid_dict=cur_forbid_dict, preference_dict=cur_preference_dict)

            ''' PASS down the update process to children distributions '''
            for kk in self.unary_children:
                self.unary_children[kk].update_by_accum_loss(lr=lr, n_samples=n_samples)

            if self.depth == 1:
                for kk in self.binary_children_left_1:
                    for cur_dist in self.binary_children_left_1[kk] + self.binary_children_right_1[kk]:
                        cur_dist.update_by_accum_loss(lr=lr, n_samples=n_samples)
            else:
                for kk in self.binary_children_left_1:
                    for cur_dist in self.binary_children_left_1[kk] + self.binary_children_left_2[kk] + self.binary_children_left_3[kk]:
                        cur_dist.update_by_accum_loss(lr=lr, n_samples=n_samples)
                    for cur_dist in self.binary_children_right_1[kk] + self.binary_children_right_2[kk] + self.binary_children_right_3[kk]:
                        cur_dist.update_by_accum_loss(lr=lr, n_samples=n_samples)

    ''' THIS function should only be called for the root node '''
    def update_sampling_dists_by_sampled_dicts(self, oper_dist_list, rewards, baseline, lr):
        n_samples = len(oper_dist_list)
        ''' ACCUMULATE operator selection & transformation selection distributions '''
        for i, cur_oper_dist_list in enumerate(oper_dist_list):
            ### If cur_oper_dist_list is an empty list, the following update will not be performed ####
            for oper_dist in cur_oper_dist_list:
                cur_reward = rewards[i]
                self.update_sampling_dists_by_sampled_dict(oper_dist, cur_reward, baseline, lr, accum=True)

        ''' UPDATE operator selection & transformation selection distributions by accumulated losses & number of samples '''
        self.update_by_accum_loss(lr=lr, n_samples=n_samples)

    def calcu_possi(self, loss_dict, cur_depth=0):
        if "oper" in loss_dict:
            uop, fop = loss_dict["uop"], loss_dict["oper"]
            cur_possi = float(self.unary_sampling_dist.params[0,uop].item()) * float((self.feat_sampling_dist[uop].params[0,fop].item()))
            return cur_possi
        else:
            oop = loss_dict["bop"] if "bop" in loss_dict else 0
            uop, gop = loss_dict["uop"], loss_dict["gop"]
            cur_possi = float(self.unary_sampling_dist.params[0,uop].item()) * float((self.grp_sampling_dist[uop].params[0,gop].item())) * float(self.operator_sampling_dist[(uop,gop)].params[0,oop].item())
            if oop == 0:
                chd_poss = self.unary_children[(uop,gop)].calcu_possi(loss_dict["chd"], cur_depth + 1)
            elif oop <= self.nn_binary_opers:
                chd_poss = self.binary_children_left_1[(uop,gop)][oop - 1].calcu_possi(loss_dict["lft_chd"], cur_depth + 1) * self.binary_children_right_1[(uop,gop)][oop - 1].calcu_possi(loss_dict["rgt_chd"], cur_depth + 1)
            elif oop <= self.nn_binary_opers * 2:
                chd_poss = self.binary_children_left_2[(uop,gop)][oop - 1 - self.nn_binary_opers].calcu_possi(loss_dict["lft_chd"], cur_depth + 1) * self.binary_children_right_2[(uop,gop)][oop - self.nn_binary_opers - 1].calcu_possi(loss_dict["rgt_chd"], cur_depth + 1)
            else:
                chd_poss = self.binary_children_left_3[(uop, gop)][oop - 1 - 2 * self.nn_binary_opers].calcu_possi(
                    loss_dict["lft_chd"], cur_depth + 1) * self.binary_children_right_3[(uop, gop)][
                               oop - 2 * self.nn_binary_opers - 1].calcu_possi(loss_dict["rgt_chd"], cur_depth + 1)
            return cur_possi * chd_poss

    ''' COLLECT distribution parameters '''
    def _collect_params(self):
        if self.depth == 2:
            cur_params = {"uop": self.unary_sampling_dist.params.detach().cpu().numpy(), "fop": [cur_feat_dist.params.detach().cpu().numpy() for cur_feat_dist in self.feat_sampling_dist]}
        else:
            cur_params = {"uop": self.unary_sampling_dist.params.detach().cpu().numpy(), "gop": [cur_gop_dist.params.detach().cpu().numpy() for cur_gop_dist in self.grp_sampling_dist], "oop": {cur_oper: self.operator_sampling_dist[cur_oper].params.detach().cpu().numpy() for cur_oper in self.operator_sampling_dist}, "unary_children": {cur_oper: self.unary_children[cur_oper]._collect_params() for cur_oper in self.unary_children}}
            if self.depth == 1:
                lft_children_params = {}
                rgt_children_params = {}
                for cur_oper in self.binary_children_left_1:
                    lft_children_params[cur_oper] = [cur_child._collect_params() for cur_child in self.binary_children_left_1[cur_oper]]
                    rgt_children_params[cur_oper] = [cur_child._collect_params() for cur_child in self.binary_children_right_1[cur_oper]]
                cur_params["lft_children"] = lft_children_params
                cur_params["rgt_children"] = rgt_children_params
            else:
                lft_children_params_1, rgt_children_params_1 = {}, {}
                lft_children_params_2, rgt_children_params_2 = {}, {}
                lft_children_params_3, rgt_children_params_3 = {}, {}
                for cur_oper in self.binary_children_left_1:
                    lft_children_params_1[cur_oper] = [cur_child._collect_params() for cur_child in
                                                     self.binary_children_left_1[cur_oper]]
                    rgt_children_params_1[cur_oper] = [cur_child._collect_params() for cur_child in
                                                     self.binary_children_right_1[cur_oper]]
                    lft_children_params_2[cur_oper] = [cur_child._collect_params() for cur_child in
                                                       self.binary_children_left_2[cur_oper]]
                    rgt_children_params_2[cur_oper] = [cur_child._collect_params() for cur_child in
                                                       self.binary_children_right_2[cur_oper]]
                    lft_children_params_3[cur_oper] = [cur_child._collect_params() for cur_child in
                                                       self.binary_children_left_3[cur_oper]]
                    rgt_children_params_3[cur_oper] = [cur_child._collect_params() for cur_child in
                                                       self.binary_children_right_3[cur_oper]]
                cur_params["lft_children_1"], cur_params["rgt_children_1"] = lft_children_params_1, rgt_children_params_1
                cur_params["lft_children_2"], cur_params["rgt_children_2"] = lft_children_params_2, rgt_children_params_2
                cur_params["lft_children_3"], cur_params["rgt_children_3"] = lft_children_params_3, rgt_children_params_3
        return cur_params

    ''' COLLECT distribution parameters and save to the file path '''
    def collect_params_and_save(self, file_name):
        tot_params = self._collect_params()
        np.save(file_name, tot_params)
